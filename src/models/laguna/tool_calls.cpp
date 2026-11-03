#include "src/models/laguna/tool_calls.hpp"

#include <initializer_list>

namespace gufo::models::laguna {
namespace {

std::string_view TrimAscii(std::string_view s) {
  const auto space = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  };
  while (!s.empty() && space(s.front()))
    s.remove_prefix(1);
  while (!s.empty() && space(s.back()))
    s.remove_suffix(1);
  return s;
}

std::size_t FindFirst(std::string_view text,
                      std::initializer_list<std::string_view> needles) {
  std::size_t best = std::string_view::npos;
  for (const auto needle : needles) {
    const auto found = text.find(needle);
    if (found != std::string_view::npos && found < best)
      best = found;
  }
  return best;
}

// Holds back a trailing partial closing tag. A streamed value must never
// carry half of "</arg_value>", or the client assembles corrupt arguments.
std::string_view BeforePartialDelimiter(std::string_view value,
                                        std::string_view delimiter) {
  for (auto size = std::min(value.size(), delimiter.size()); size > 0; --size)
    if (value.ends_with(delimiter.substr(0, size)))
      return value.substr(0, value.size() - size);
  return value;
}

}  // namespace

LagunaToolCallScan ScanLagunaToolCall(std::string_view text) {
  LagunaToolCallScan result;
  if (!text.starts_with(LagunaToolSyntax::kCallOpen))
    return result;
  result.matched = true;
  const auto offset = [&](std::string_view rest) {
    return static_cast<std::size_t>(rest.data() - text.data());
  };
  std::string_view body = text.substr(LagunaToolSyntax::kCallOpen.size());
  std::size_t progress = LagunaToolSyntax::kCallOpen.size();
  result.progress = progress;
  result.end = progress;

  LagunaToolCallSyntax call;
  // The function name precedes the first argument tag.
  const auto first_key = body.find(LagunaToolSyntax::kArgKeyOpen);
  const auto first_close = body.find(LagunaToolSyntax::kCallClose);
  std::size_t name_end = body.size();
  if (first_key != std::string_view::npos &&
      (first_close == std::string_view::npos || first_key < first_close)) {
    name_end = first_key;
  } else if (first_close != std::string_view::npos) {
    name_end = first_close;
  }
  call.name = std::string(TrimAscii(body.substr(0, name_end)));
  body.remove_prefix(name_end);
  progress = offset(body);

  bool valid = !call.name.empty();
  while (valid && body.starts_with(LagunaToolSyntax::kArgKeyOpen)) {
    body.remove_prefix(LagunaToolSyntax::kArgKeyOpen.size());
    const auto key_close = body.find(LagunaToolSyntax::kArgKeyClose);
    if (key_close == std::string_view::npos) {
      valid = false;
      break;
    }
    const std::string name(body.substr(0, key_close));
    body.remove_prefix(key_close + LagunaToolSyntax::kArgKeyClose.size());
    if (!body.starts_with(LagunaToolSyntax::kArgValueOpen)) {
      valid = false;
      break;
    }
    body.remove_prefix(LagunaToolSyntax::kArgValueOpen.size());
    // The value ends at the first of the closing tag, the next key or the
    // next value tag. A greedy scan would swallow the next argument's key
    // when the closing tag is missing.
    const auto value_end = FindFirst(
        body, {LagunaToolSyntax::kArgValueClose, LagunaToolSyntax::kArgKeyOpen,
               LagunaToolSyntax::kArgValueOpen});
    if (value_end == std::string_view::npos) {
      call.unfinished =
          LagunaToolCallArgument{.name = name,
                                 .value = std::string(BeforePartialDelimiter(
                                     body, LagunaToolSyntax::kArgValueClose)),
                                 .progress = text.size()};
      progress = text.size();
      valid = false;
      break;
    }
    const std::string value(body.substr(0, value_end));
    body.remove_prefix(value_end);
    if (!body.starts_with(LagunaToolSyntax::kArgValueClose)) {
      // The value stopped at the next key: its closing tag was missing. The
      // call is malformed, as llama.cpp's Laguna parser treats it.
      progress = offset(body);
      valid = false;
      break;
    }
    body.remove_prefix(LagunaToolSyntax::kArgValueClose.size());
    progress = offset(body);
    call.arguments.push_back(LagunaToolCallArgument{
        .name = name, .value = value, .progress = progress});
  }
  // The model may leave whitespace between the last argument and the
  // closing tag, as llama.cpp's Laguna parser tolerates.
  const auto rest = TrimAscii(body);
  if (valid && rest.starts_with(LagunaToolSyntax::kCallClose)) {
    call.complete = true;
    progress = offset(rest) + LagunaToolSyntax::kCallClose.size();
  }
  result.progress = progress;
  result.end = progress;
  result.call = std::move(call);
  return result;
}

}  // namespace gufo::models::laguna
