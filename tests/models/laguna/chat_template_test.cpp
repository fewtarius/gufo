// Renders conversations through the Laguna chat template and compares them
// byte for byte with the output of the artifact's Jinja template
// (fixtures/chat_cases.json, produced with jinja2 and llama.cpp's tojson
// spelling).
#include "src/models/laguna/chat_template.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace {

namespace laguna = gufo::models::laguna;
namespace json = gufo::json;

laguna::ChatMessage ToMessage(const json::Value& value) {
  laguna::ChatMessage message;
  message.role = value.member_str("role");
  message.content = value.member_str("content");
  message.reasoning_content = value.member_str("reasoning_content");
  if (const auto* calls = value.find("tool_calls")) {
    for (const auto& call : calls->items()) {
      laguna::ChatToolCall native;
      const auto& function = *call.find("function");
      native.name = function.member_str("name");
      for (const auto& [key, argument] : function.find("arguments")->members()) {
        native.arguments.push_back(
            {key,
             argument.is_string() ? argument.str() : argument.tojson(),
             argument.is_string()});
      }
      message.tool_calls.push_back(std::move(native));
    }
  }
  return message;
}

laguna::ChatTool ToTool(const json::Value& value) {
  laguna::ChatTool tool;
  const auto& function = *value.find("function");
  tool.name = function.member_str("name");
  tool.description = function.member_str("description");
  tool.parameters_json = function.find("parameters")->dump();
  return tool;
}

}  // namespace

int main() {
  std::ifstream in(GUFO_LAGUNA_CHAT_CASES);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const auto cases = json::parse(buffer.str());
  int failures = 0;
  for (const auto& entry : cases.items()) {
    std::vector<laguna::ChatMessage> messages;
    for (const auto& message : entry.find("messages")->items())
      messages.push_back(ToMessage(message));
    std::vector<laguna::ChatTool> tools;
    if (const auto* list = entry.find("tools");
        list != nullptr && list->is_array()) {
      for (const auto& tool : list->items())
        tools.push_back(ToTool(tool));
    }
    laguna::ChatTemplateOptions options;
    const auto* thinking = entry.find("thinking");
    options.enable_thinking = thinking != nullptr && thinking->as_bool(false);
    options.tools = tools;
    const auto actual = laguna::RenderPrompt(messages, options);
    const auto expected = entry.member_str("expected");
    const bool ok = actual == expected;
    std::printf("%-28s %s\n", entry.member_str("name").c_str(),
                ok ? "OK" : "FAIL");
    if (!ok) {
      ++failures;
      std::printf("  expected: %s\n  actual:   %s\n",
                  json::Value(expected).dump().c_str(),
                  json::Value(actual).dump().c_str());
    }
  }
  return failures == 0 ? 0 : 1;
}
