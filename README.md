# Wikisearch Agent

A small agent to delve into Wikipedia and structure some information.

This is partially a learning exercise in LangChain/Graph + MCP, but it's also a demo of using LLMs to structure information from real-world freetext data. The point of this demo is to:

- Identify people-like entities: We're primarily trying to find real, historical people, and differentiate them from, say, fictional people or entities.
- Structure information about each entity: Name, aliases, birth date, locality of origin, gender (both assigned at birth and identity), and so on.
- Search Wikipedia to discover new such entities.
