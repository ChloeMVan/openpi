import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge # bridge allows nnx modules through jax transforms
# Flax is a neural network library for JAX and nnx is the api
# π0: uses Flax to build the transformer, vision encoder, etc.
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos( # turns timestep into embedding
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img) # PaliGemma is a dictionary hold two models, an LLM and vision encode
        # PaliGemma is a VLM but its made of an LLM (Gemma) and vision encoder (SigLIP)
        # PaliGemma.img is the vision encoder forward pass. It takes an image tensor and returns image token embeddings
        
        # Linear is a matrix multiply + bias
        
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        # for projecting the action tokens, (turns each action chunk into a token embedding)
        # [B, H, action_dim] → [B, H, width]

        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            # projects robot state into width 

            # these are called mlp because they're used like Linear + activation + Linear in the code
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        # Maps the transformer’s output back into action space.

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        '''
        high level: Embed all observation context (images + language) 
        into a single prefix token sequence.
        '''
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            # vision encoder on the images from observation
            # runs through sigLIP model
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]
            # True = barrier: prefix tokens CANNOT attend past this point
            # This ensures images/text don't attend to state/actions

        action_tokens = self.action_in_proj(noisy_actions) # 1 Linear layer to convert noisy actions into tokens

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb) # Linear layer
            time_emb = nnx.swish(time_emb) # activation layer
            time_emb = self.time_mlp_out(time_emb) # linear layer
            time_emb = nnx.swish(time_emb) # activation layer
            action_expert_tokens = action_tokens 
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            # time_emb is repeated to match the action sequence length (action_horizon)

            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens) # linear layer
            action_time_tokens = nnx.swish(action_time_tokens) # activation layer
            action_time_tokens = self.action_time_mlp_out(action_time_tokens) # linear layer
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10, # changed from 10
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        with jax.named_scope("preprocess_observation"):
            # jax.debug.print("post processing observation")
            ob_dict = observation.to_dict()
            # for key,val in ob_dict.items():
                # jax.debug.print("\tkey: {} {}", type(key), 0)
                # jax.debug.print("\tvalue: {} {}", type(val), 0)


            # observation contains 
            observation = _model.preprocess_observation(None, observation, train=False)
            '''Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
            filling in a default image mask (if necessary).
            return Observation(
                images=out_images,
                image_masks=out_masks,
                state=observation.state,
                tokenized_prompt=observation.tokenized_prompt,
                tokenized_prompt_mask=observation.tokenized_prompt_mask,
                token_ar_mask=observation.token_ar_mask,
                token_loss_mask=observation.token_loss_mask,
            )
            '''
            # jax.debug.print("post processing observation")
            ob_dict = observation.to_dict()
            # for key,val in ob_dict.items():
            #     jax.debug.print("\tkey: {} {}", type(key), 0)
            #     jax.debug.print("\tvalue: {} {}", type(val), 0)


        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        with jax.named_scope("init_noise_dt"):
            dt = -1.0 / num_steps
            batch_size = observation.state.shape[0]
            if noise is None:
                noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        with jax.named_scope("embed_prefix"):
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            # Creates a full attention mask
            # prefix tokens contains both embeddings of images and prompt embeddings
        
        
        with jax.named_scope("prefix_kv_cache_llm"):
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1 
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
            # jax.debug.print("KV_cache shapes: {} {}", kv_cache[0].shape, kv_cache[1].shape)
            # jax.debug.print("KV_cache type: {} {}\n", type(kv_cache[0]), type(kv_cache[1]))
            '''
            PaliGemma (pi0 paper): {width=2048, depth=18, mlp dim=16,384, num heads=18, num kv heads=1, head dim=256}

            [layers, batch, sequence_length, num_kv_heads, head_dim]

            KV_cache shapes: (Array(18, dtype=int32), Array(1, dtype=int32), Array(816, dtype=int32), 
                Array(1, dtype=int32), Array(256, dtype=int32)) (Array(18, dtype=int32), 
                Array(1, dtype=int32), Array(816, dtype=int32), Array(1, dtype=int32), 
                Array(256, dtype=int32))
                
            KV_cache type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> 
                <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>
            '''

            # To see actual values:
            # jax.debug.print("KV_cache[0][0,0,:5,0,:5]: {}", kv_cache[0][0,0,:5,0,:5])
            # jax.debug.print("KV_cache[1][0,0,:5,0,:5]: {}", kv_cache[1][0,0,:5,0,:5])

        def step(carry):
            x_t, time = carry
            timestep_print = jnp.broadcast_to(time, batch_size)
            # jax.debug.print("=== Step at time={:.3f} ===", time)
            # jax.debug.print("Current KV cache K shape: {}", kv_cache[0].shape)
            # jax.debug.print("Current KV cache V shape: {}", kv_cache[1].shape)
            with jax.named_scope("embed_suffix"):
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation, x_t, timestep_print
                )
                # jax.debug.print("timestep value: {}, shape: {}, dtype: {}", 
                # timestep_print, timestep_print.shape, timestep_print.dtype)

                # The time step value is broadcasted to match the batch size so that every
                #  example in the batch gets the same timestep.

            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            # suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            # prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            # full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            # assert full_attn_mask.shape == (
            #     batch_size,
            #     suffix_tokens.shape[1],
            #     prefix_tokens.shape[1] + suffix_tokens.shape[1],
            # )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            # positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            with jax.named_scope("build_attention_masks"):
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
                # jax.debug.print("full_attn_mask shape: {}", full_attn_mask.shape)
                # jax.debug.print("full_attn_mask: {}", full_attn_mask)
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                # jax.debug.print("suffix positions: {}", positions[0])

            with jax.named_scope("suffix_llm_forward"):
                (prefix_out, suffix_out), returned_kv_cache = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
            
            # if kv_cache is not None:
            #     # Look at the LAST 51 positions in cache
            #     last_positions = jnp.arange(kv_cache[0].shape[2] - 51, kv_cache[0].shape[2])
            #     jax.debug.print("Cache positions being written to: {}", last_positions)

            # jax.debug.print("Returned_kv_cache seq_len: {}", returned_kv_cache[0].shape[2])

            assert prefix_out is None

            with jax.named_scope("action_out_proj"):
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            with jax.named_scope("euler_update"):
                return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            # Continue while time >= -dt/2 (approximately time > 0)
            return time >= -dt / 2

        with jax.named_scope("flow_while_loop"):
            x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
            
        return x_0