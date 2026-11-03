#ifndef GUFO_MODELS_LAGUNA_TOOL_CALLS_HPP_
#define GUFO_MODELS_LAGUNA_TOOL_CALLS_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::laguna {

/// The literals that frame a native Laguna tool call. The chat template
/// renders historical calls with these, and ScanLagunaToolCall reads the same
/// bytes back, so the wire format cannot change in one direction only.
struct LagunaToolSyntax {
  static constexpr std::string_view kCallOpen = "<tool_call>";
  static constexpr std::string_view kCallClose = "</tool_call>";
  static constexpr std::string_view kArgKeyOpen = "<arg_key>";
  static constexpr std::string_view kArgKeyClose = "</arg_key>";
  static constexpr std::string_view kArgValueOpen = "<arg_value>";
  static constexpr std::string_view kArgValueClose = "</arg_value>";
};

/// One argument exactly as the model wrote it, before the API surface decides
/// whether it is a string or JSON.
struct LagunaToolCallArgument {
  std::string name;   ///< Spelled between "<arg_key>" and "</arg_key>".
  std::string value;  ///< The bytes between "<arg_value>" and its close.
  /// Bytes of the scanned text consumed through this argument. Recovery
  /// resumes here when the API surface rejects this argument.
  std::size_t progress{0};
};

/// One call exactly as written, before schema interpretation.
struct LagunaToolCallSyntax {
  std::string name;
  std::vector<LagunaToolCallArgument> arguments;
  /// The argument whose value ran to the end of the scan without a closing
  /// tag.
  std::optional<LagunaToolCallArgument> unfinished;
  /// The closing call tag and every argument delimiter were present.
  bool complete{false};
};

/// One scan step over a response. The caller keeps quote state and schema
/// interpretation; this result carries only the dialect bytes.
struct LagunaToolCallScan {
  bool matched{false};  ///< `text` begins with LagunaToolSyntax::kCallOpen.
  /// Bytes of `text` consumed before scanning resumes when the call is not
  /// accepted into the API output.
  std::size_t progress{0};
  /// Bytes of `text` belonging to a complete envelope.
  std::size_t end{0};
  std::optional<LagunaToolCallSyntax> call;  ///< Absent when nothing parsed.
};

/// Scans the Laguna call beginning at `text[0]`, if any. Schema
/// interpretation, argument coercion and OpenAI shaping stay with the API
/// surface.
[[nodiscard]] LagunaToolCallScan ScanLagunaToolCall(std::string_view text);

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_TOOL_CALLS_HPP_
