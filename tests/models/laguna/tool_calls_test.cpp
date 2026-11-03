// Scans native Laguna tool calls and checks the dialect's boundaries.
#include "src/models/laguna/tool_calls.hpp"

#include <cstdio>
#include <string>
#include <string_view>

namespace {

namespace laguna = gufo::models::laguna;

int failures = 0;

void Check(bool ok, const char* label) {
  std::printf("%-46s %s\n", label, ok ? "OK" : "FAIL");
  if (!ok)
    ++failures;
}

}  // namespace

int main() {
  using laguna::LagunaToolSyntax;

  {
    const std::string text =
        "<tool_call>skill_operations<arg_key>operation</arg_key>"
        "<arg_value>load</arg_value><arg_key>name</arg_key>"
        "<arg_value>audit-codebase</arg_value></tool_call>";
    const auto scan = laguna::ScanLagunaToolCall(text);
    Check(scan.matched && scan.call.has_value(), "scan matched");
    if (!scan.call)
      return 1;
    const auto& call = *scan.call;
    Check(call.complete, "call complete");
    Check(call.name == "skill_operations", "call name");
    Check(call.arguments.size() == 2, "argument count");
    if (call.arguments.size() == 2) {
      Check(call.arguments[0].name == "operation" &&
                call.arguments[0].value == "load",
            "first argument");
      Check(call.arguments[1].name == "name" &&
                call.arguments[1].value == "audit-codebase",
            "second argument");
    }
    Check(scan.end == text.size(), "scan consumed the call");
  }

  {
    // A value stops at the first of the closing tag, the next key or the
    // next value tag. A greedy scan swallows the next key.
    const std::string text =
        "<tool_call>f<arg_key>good</arg_key><arg_value>ok</arg_value>"
        "<arg_key>operation</arg_key><arg_value>read"
        "<arg_key>path</arg_key><arg_value>/tmp/x</arg_value></tool_call>";
    const auto scan = laguna::ScanLagunaToolCall(text);
    Check(scan.matched && scan.call.has_value(), "unterminated value matched");
    if (scan.call) {
      const auto& call = *scan.call;
      Check(!call.complete, "unterminated value is not complete");
      Check(call.arguments.size() == 1, "the well-formed argument survived");
      if (call.arguments.size() == 1) {
        Check(call.arguments[0].name == "good" &&
                  call.arguments[0].value == "ok",
              "the well-formed argument is intact");
      }
    }
  }

  {
    // The first argument's closing tag is missing: the call fails to parse
    // and no argument is accepted from it.
    const std::string text =
        "<tool_call>f<arg_key>operation</arg_key><arg_value>read"
        "<arg_key>path</arg_key><arg_value>/tmp/x</arg_value></tool_call>";
    const auto scan = laguna::ScanLagunaToolCall(text);
    Check(scan.matched && scan.call.has_value(), "malformed value matched");
    if (scan.call) {
      Check(!scan.call->complete, "malformed value is not complete");
      Check(scan.call->arguments.empty(),
            "malformed value contributes no argument");
      Check(!scan.call->unfinished.has_value(),
            "malformed value is not an unfinished value");
    }
  }

  {
    // A value that runs to the end of the scan is unfinished, not corrupt.
    const std::string text = "<tool_call>f<arg_key>k</arg_key><arg_value>v";
    const auto scan = laguna::ScanLagunaToolCall(text);
    Check(scan.matched && scan.call.has_value(), "open value matched");
    if (scan.call) {
      const auto& call = *scan.call;
      Check(!call.complete, "open value is not complete");
      Check(call.unfinished.has_value(), "open value is unfinished");
      if (call.unfinished) {
        Check(call.unfinished->name == "k", "unfinished argument name");
        Check(call.unfinished->value == "v", "unfinished argument value");
      }
    }
  }

  {
    // A half-written closing tag must not reach the streamed value.
    const std::string text =
        "<tool_call>f<arg_key>k</arg_key><arg_value>load</arg_value";
    const auto scan = laguna::ScanLagunaToolCall(text);
    Check(scan.matched && scan.call.has_value(), "half close matched");
    if (scan.call) {
      const auto& call = *scan.call;
      Check(call.unfinished.has_value(), "half close is unfinished");
      if (call.unfinished)
        Check(call.unfinished->value == "load",
              "half close holds back the partial tag");
    }
  }

  {
    // Whitespace before the closing tag is tolerated.
    const std::string text =
        "<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value>   "
        "</tool_call>";
    const auto scan = laguna::ScanLagunaToolCall(text);
    Check(scan.matched && scan.call.has_value(), "spaced close matched");
    if (scan.call) {
      Check(scan.call->complete, "spaced close is complete");
      Check(scan.end == text.size(), "spaced close consumed the call");
    }
  }

  {
    // A call with no arguments ends at the closing tag.
    const std::string text = "<tool_call>f</tool_call>";
    const auto scan = laguna::ScanLagunaToolCall(text);
    Check(scan.matched && scan.call.has_value(), "empty call matched");
    if (scan.call) {
      Check(scan.call->complete, "empty call is complete");
      Check(scan.call->name == "f", "empty call name");
      Check(scan.call->arguments.empty(), "empty call has no arguments");
    }
  }

  {
    const std::string text = "not a call";
    const auto scan = laguna::ScanLagunaToolCall(text);
    Check(!scan.matched, "prose is not a call");
  }

  return failures == 0 ? 0 : 1;
}
