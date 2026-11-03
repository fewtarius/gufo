#ifndef GUFO_MODELS_LAGUNA_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_LAGUNA_CHAT_TEMPLATE_HPP_

#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::laguna {

inline constexpr std::string_view kBosText = "\u3008|EOS|\u3009";
inline constexpr std::string_view kAssistantOpen = "<assistant>";
inline constexpr std::string_view kAssistantClose = "</assistant>";
inline constexpr std::string_view kThinkOpen = "<think>";
inline constexpr std::string_view kThinkClose = "</think>";
inline constexpr std::string_view kDefaultSystem =
    "You are a helpful, conversationally-fluent assistant made by Poolside. "
    "You are here to be helpful to users through natural language "
    "conversations.";

/// One argument of a native tool call.
struct ChatToolCallArgument {
  std::string name;
  std::string value;
  /// True when the value is text; false when it is a JSON document.
  bool is_string{true};
};

/// One native tool call.
struct ChatToolCall {
  std::string name;
  std::vector<ChatToolCallArgument> arguments;
};

struct ChatMessage {
  std::string role;  // system | developer | user | assistant | tool
  std::string content;
  std::string reasoning_content;
  std::vector<ChatToolCall> tool_calls;  // assistant only
};

/// One advertised tool, as the client sent it.
struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
};

struct ChatTemplateOptions {
  bool enable_thinking{false};
  std::vector<ChatTool> tools;
};

// Mirrors the GGUF Jinja template for system/user/assistant turns with
// add_generation_prompt.
[[nodiscard]] std::string RenderPrompt(const std::vector<ChatMessage>& messages,
                                       const ChatTemplateOptions& options = {});

// Convenience: system + single user turn.
[[nodiscard]] std::string RenderPrompt(std::string_view system,
                                       std::string_view user,
                                       const ChatTemplateOptions& options = {});

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_CHAT_TEMPLATE_HPP_
