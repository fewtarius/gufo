# Tool dialect scan offsets

Tool Dialect (`src/ext/text.hpp`): the scanned call carries the offsets the
API surface needs to recover from a rejected call.
- `ScannedToolArgument` gains `progress`, the bytes of the scanned text
  consumed through that argument. When the API surface rejects the argument,
  the scan resumes there rather than after the whole call.
- `ScannedToolCall` gains `begin`, where the call starts in the scanned text,
  `progress`, the bytes consumed through the call, and `end`, where the call's
  own markup ends. An unfinished call reports no `end`.

The offsets are measured in the text the scanner was given. A dialect's
scanner fills them in; the DeepSeek and Laguna scanners do. Additive;
extensions that aggregate-initialize these structs positionally keep the
defaults.
