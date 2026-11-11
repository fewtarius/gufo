#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/laguna/probe.hpp"
#include "src/models/laguna/tool_calls.hpp"

namespace gufo::models::laguna {
namespace {

/// Adapts the Laguna scanner to the shared dialect contract.
///
/// The Laguna scanner reports one call at a time; the shared contract
/// reports every call in the scanned text, so the adapter scans each opener
/// the dialect declares.
///
/// The contract measures its offsets in the text the scanner was given, so
/// the adapter rebases the scanner's call-relative offsets onto the call's
/// position in that text.
void ScanLagunaDialect(std::string_view text,
                       std::span<const std::string_view> markers,
                       std::vector<ext::ScannedToolCall>* calls,
                       std::vector<std::pair<std::size_t, std::size_t>>* spans,
                       bool partial) {
  if (calls == nullptr) {
    return;
  }
  std::size_t offset = 0;
  while (offset < text.size()) {
    // The opener can sit anywhere in the text, and the text before it is
    // prose. Find the earliest opener the dialect declares.
    std::size_t begin = std::string_view::npos;
    std::size_t opener = 0;
    for (const auto marker : markers) {
      const auto found = text.find(marker, offset);
      if (found != std::string_view::npos && found < begin) {
        begin = found;
        opener = marker.size();
      }
    }
    if (begin == std::string_view::npos) {
      break;
    }
    const auto scanned = ScanLagunaToolCall(text.substr(begin));
    if (!scanned.matched) {
      offset = begin + opener;
      continue;
    }
    ext::ScannedToolCall call;
    call.name = scanned.call ? scanned.call->name : std::string{};
    if (scanned.call) {
      for (const auto& argument : scanned.call->arguments) {
        call.arguments.push_back(
            ext::ScannedToolArgument{.name = argument.name,
                                     .value = argument.value,
                                     .is_string = true,
                                     .progress = begin + argument.progress});
      }
      if (scanned.call->unfinished) {
        call.unfinished = ext::ScannedToolArgument{
            .name = scanned.call->unfinished->name,
            .value = scanned.call->unfinished->value,
            .is_string = true,
            .progress = begin + scanned.call->unfinished->progress};
      }
    }
    call.complete = scanned.call ? scanned.call->complete : false;
    // The shared contract marks a call valid when the scanner parsed it
    // completely; the API surface still checks the schema.
    call.valid = call.complete;
    call.begin = begin;
    call.progress = begin + scanned.progress;
    // An unfinished call owns no closed envelope, so it has no end.
    call.end = call.complete ? begin + scanned.end : 0;
    calls->push_back(std::move(call));
    // The API surface decides which calls it accepts, and a rejected call
    // resumes the scan inside itself. Report every opener so that decision
    // stays with the API surface.
    offset = begin + opener;
  }
  (void)spans;
  (void)partial;
}

}  // namespace

void RegisterToolDialect() {
  // The Laguna envelope names the function directly after the opener and tags
  // each argument as <arg_key>KEY</arg_key><arg_value>VALUE</arg_value>. It is
  // a tagged-argument grammar with Laguna's own tag text, so it is declared as
  // a provider syntax rather than a built-in ToolFormat.
  static const sampling::JsonConstraint::ToolSyntax kSyntax{
      .open = std::string(LagunaToolSyntax::kCallOpen),
      .name_prefix = "",
      .name_suffix = "",
      .close = std::string(LagunaToolSyntax::kCallClose),
      // The name ends at the first argument key or the closing tag.
      .name_delimiters = "<",
      .tagged =
          sampling::JsonConstraint::TaggedArguments{
              .parameter_open = std::string(LagunaToolSyntax::kArgKeyOpen),
              // The key ends where its close and the value's open meet.
              .raw_value_open = std::string(LagunaToolSyntax::kArgKeyClose) +
                                std::string(LagunaToolSyntax::kArgValueOpen),
              .typed_value_open = std::string(LagunaToolSyntax::kArgKeyClose) +
                                  std::string(LagunaToolSyntax::kArgValueOpen),
              .value_close = std::string(LagunaToolSyntax::kArgValueClose),
              .parameter_separator = "",
          },
      .reasoning_header = true,
  };
  ext::RegisterToolDialect({
      .markers = {LagunaToolSyntax::kCallOpen},
      // Laguna names the function directly after the opener, so the opener
      // itself is the native header.
      .reasoning_markers = {LagunaToolSyntax::kCallOpen},
      .closers =
          {
              LagunaToolSyntax::kArgValueClose,
              LagunaToolSyntax::kCallClose,
              // The client's envelope, whose block rules already strip it
              // whole.
              "</invoke>",
          },
      .scan_tool_calls = &ScanLagunaDialect,
      .id = "laguna",
      .syntax = kSyntax,
  });
}

}  // namespace gufo::models::laguna

void laguna_register(gufo::ext::Registry&) {
  gufo::models::laguna::RegisterToolDialect();
}
