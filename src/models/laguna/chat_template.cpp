#include "src/models/laguna/chat_template.hpp"

#include <exception>
#include <gufo/ext/json.hpp>

namespace gufo::models::laguna {
namespace {

std::string_view TrimRight(std::string_view s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' ||
                        s.back() == '\t' || s.back() == '\r')) {
    s.remove_suffix(1);
  }
  return s;
}

bool IsBlank(std::string_view s) {
  return TrimRight(s).empty() ||
         s.find_first_not_of(" \n\t\r") == std::string_view::npos;
}

// A JSON text as Python's json.dumps writes it; text that does not parse
// is kept as the client wrote it.
std::string PythonJson(std::string_view text) {
  try {
    return json::parse(text).tojson();
  } catch (const std::exception&) {
    return std::string(text);
  }
}

// The tool object the template's `tool | tojson` writes.
std::string ToolJson(const ChatTool& tool) {
  std::string out =
      "{\"type\": \"function\", \"function\": {\"name\": " +
      json::Value(tool.name).tojson() +
      ", \"description\": " + json::Value(tool.description).tojson() +
      ", \"parameters\": " +
      PythonJson(tool.parameters_json.empty() ? "{}" : tool.parameters_json) +
      "}}";
  return out;
}

void AppendTools(std::string& out, const std::vector<ChatTool>& tools) {
  out += "### Tools\n\n";
  out += "You may call functions to assist with the user query.\n";
  out += "All available function signatures are listed below:\n";
  out += "<available_tools>\n";
  for (const auto& tool : tools) {
    out += ToolJson(tool);
    out += "\n";
  }
  out += "</available_tools>";
}

void AppendToolCalls(std::string& out, const std::vector<ChatToolCall>& calls) {
  for (const auto& call : calls) {
    out += "<tool_call>";
    out += call.name;
    for (const auto& argument : call.arguments) {
      out += "<arg_key>";
      out += argument.name;
      out += "</arg_key><arg_value>";
      out += argument.is_string ? argument.value : PythonJson(argument.value);
      out += "</arg_value>";
    }
    out += "</tool_call>";
  }
}

}  // namespace

std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                         const ChatTemplateOptions& options) {
  std::string out(kBosText);
  std::size_t first = 0;
  std::string_view system = kDefaultSystem;
  if (!messages.empty() && messages[0].role == "system") {
    system = messages[0].content;
    first = 1;
  }
  const bool has_sys = !IsBlank(system);
  const bool has_tools = !options.tools.empty();
  if (has_sys || has_tools || options.enable_thinking) {
    out += "<system>";
    if (has_sys) {
      out += TrimRight(system);
      if (has_tools)
        out += "\n\n";
    }
    if (has_tools)
      AppendTools(out, options.tools);
    out += "</system>\n";
  }
  for (std::size_t i = first; i < messages.size(); ++i) {
    const auto& m = messages[i];
    if (m.role == "user") {
      out += "<user>" + m.content + "</user>\n";
    } else if (m.role == "assistant") {
      out += kAssistantOpen;
      if (options.enable_thinking) {
        out += kThinkOpen;
        out += m.reasoning_content;
        out += kThinkClose;
      } else {
        out += kThinkClose;
      }
      out += m.content;
      AppendToolCalls(out, m.tool_calls);
      out += kAssistantClose;
      out += "\n";
    } else if (m.role == "tool") {
      out += "<tool_response>" + m.content + "</tool_response>\n";
    } else if (m.role == "system") {
      out += "<system>" + m.content + "</system>\n";
    }
  }
  out += kAssistantOpen;
  out += options.enable_thinking ? kThinkOpen : kThinkClose;
  return out;
}

std::string RenderPrompt(std::string_view system, std::string_view user,
                         const ChatTemplateOptions& options) {
  std::vector<ChatMessage> messages;
  if (!system.empty()) {
    messages.push_back({"system", std::string(system), "", {}});
  }
  messages.push_back({"user", std::string(user), "", {}});
  return RenderPrompt(messages, options);
}

}  // namespace gufo::models::laguna
