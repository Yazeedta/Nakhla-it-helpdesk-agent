
import asyncio
import os
import sys

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


load_dotenv()

# Start the existing MCP server as a Python subprocess
server_params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "services.helpdesk_mcp"],
    env=os.environ.copy()
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover available MCP tools
            available = await session.list_tools()

            print("Available MCP tools:")

            for tool in available.tools:
                print("-", tool.name)

            # Test a read-only tool
            result = await session.call_tool(
                "get_service_status",
                arguments={}
            )

            if result.isError:
                raise RuntimeError("MCP tool returned an error")

            print("\nService status:")

            for content in result.content:
                if content.type == "text":
                    print(content.text)


if __name__ == "__main__":
    asyncio.run(main())

