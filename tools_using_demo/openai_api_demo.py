import argparse
import json

from colorama import Fore, init
from loguru import logger
from openai import OpenAI
from tool_register import dispatch_tool, get_tools

parser = argparse.ArgumentParser()
parser.add_argument("--key", type=str)
args = parser.parse_args()

init(autoreset=True)
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=args.key)

tools = get_tools()


def run_conversation(query: str, stream=False, tools=None, max_retry=5):
    params = dict(model="chatglm3", messages=[{"role": "user", "content": query}], stream=stream)
    if tools:
        params["tools"] = tools
    # logger.info(f'{tools.keys()} {type(tools)}')
    # for k, v in tools.items():
    #     logger.info(f'{k = }\n{v = }')
    response = client.chat.completions.create(**params)

    for _ in range(max_retry):
        if not stream:
            if response.choices[0].message.function_call:
                function_call = response.choices[0].message.function_call
                logger.info(f"Function Call Response: {function_call.model_dump()}")

                function_args = json.loads(function_call.arguments)
                tool_response = dispatch_tool(function_call.name, function_args)
                logger.info(f"Tool Call Response: {tool_response}")

                params["messages"].append(response.choices[0].message)
                params["messages"].append(
                    {
                        "role": "function",
                        "name": function_call.name,
                        "content": tool_response,  # 调用函数返回结果
                    }
                )
            else:
                reply = response.choices[0].message.content
                logger.info(f"Final Reply: \n{reply}")
                return

        else:
            output = ""
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(Fore.BLUE + content, end="", flush=True)
                output += content

                if chunk.choices[0].finish_reason == "stop":
                    return

                elif chunk.choices[0].finish_reason == "function_call":
                    print("\n")

                    function_call = chunk.choices[0].delta.function_call
                    function_args = json.loads(function_call.arguments)
                    tool_response = dispatch_tool(function_call.name, function_args)
                    logger.info(
                        f"{function_call.name = }, {function_args = }, "
                        f"Tool Call Response: {tool_response}"
                    )

                    params["messages"].append(
                        {
                            "role": "assistant",
                            "content": output
                        }
                    )
                    params["messages"].append(
                        {
                            "role": "function",
                            "name": function_call.name,
                            "content": tool_response,
                        }
                    )

                    break

        response = client.chat.completions.create(**params)


if __name__ == "__main__":
    # query = "你是谁? 用英文回答"
    # query = "who are you?"
    # run_conversation(query, stream=True)

    # logger.info("\n=========== next conversation ===========")

    query = "帮我查询上海的天气怎么样, 用英文回答"
    run_conversation(query, tools=tools, stream=True)
