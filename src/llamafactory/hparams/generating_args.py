# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from transformers import GenerationConfig


@dataclass
class GeneratingArguments:
    r"""Arguments pertaining to specify the decoding parameters."""

    # ---- custom decoding helpers ----
    enable_tag_debug: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, when the last generated token equals `tag_debug_trigger_token`, restrict the next token "
                "to `tag_debug_candidate_tokens`."
            )
        },
    )
    tag_debug_trigger_token: str = field(
        default="<TAG>",
        metadata={"help": "Trigger token that activates tag routing."},
    )
    tag_debug_candidate_tokens: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Candidate tokens to follow the trigger. Use commas to separate multiple tokens."},
    )
    tag_debug_conditional_phrase: Optional[str] = field(
        default=None,
        metadata={
            "help": "If set, and the generated text immediately before trigger matches this phrase, use override tokens."
        },
    )
    tag_debug_override_tokens: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Override candidate tokens when conditional phrase matches. Use commas to separate tokens."},
    )
    tag_debug_force_eos_after_candidate: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, once a tag_debug_candidate token is generated, force the next token to eos to stop decoding."
            )
        },
    )
    skip_eot_id: bool = field(
        default=False,
        metadata={"help": "If True, remove <|eot_id|> after decoding while keeping other special tokens."},
    )

    # ---- standard generation configs ----
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."},
    )
    temperature: float = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: float = field(
        default=0.7,
        metadata={
            "help": (
                "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
            )
        },
    )
    top_k: int = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."},
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."},
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."},
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "Whether or not to remove special tokens in the decoding."},
    )

    def __post_init__(self):
        if isinstance(self.tag_debug_candidate_tokens, str):
            self.tag_debug_candidate_tokens = [
                token.strip() for token in self.tag_debug_candidate_tokens.split(",") if token.strip()
            ]

    def to_dict(self, obey_generation_config: bool = False) -> dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)

        if obey_generation_config:
            generation_config = GenerationConfig()
            for key in list(args.keys()):
                if not hasattr(generation_config, key):
                    args.pop(key)

        return args
