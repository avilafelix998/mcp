# agent.py
# Este archivo es el cliente que consume las herramientas del servidor MCP.
# Contiene un agente de IA (usando LangChain y Gemini) que entiende una tarea,
# planifica los pasos y utiliza las herramientas del servidor para completarla.

import os
import asyncio
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

async def main():
    # Cargar la clave de API desde el archivo .env
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("No se encontró la GOOGLE_API_KEY. Asegúrate de crear un archivo .env con tu clave.")

    # Configurar el cliente MCP para conectar con nuestro servidor.
    client = MultiServerMCPClient({
        "campaign_tools": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",
        }
    })

    print(" Conectando al servidor MCP para obtener las herramientas disponibles...")
    tools = await client.get_tools()
    print(f"  Herramientas obtenidas exitosamente: {[tool.name for tool in tools]}")

    # Inicializar el modelo LLM que potenciará al agente.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    # 4. Crear el prompt para el agente.
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto Community Manager. Tu tarea es generar y publicar una campaña completa en redes sociales usando las herramientas disponibles. Debes seguir las instrucciones del usuario al pie de la letra."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Crear el agente y el ejecutor.
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 6. Interacción con el usuario
    print("\n Asistente de Campañas para Redes Sociales Activado ")
    print("-------------------------------------------------------")

    producto = input("Por favor, ingresa el nombre del producto: ")
    publico = input("Ahora, describe el público objetivo (ej. 'jóvenes gamers', 'profesionales de la tecnología'): ")

    # Crear el input final para el agente
    user_input = f"""
    Por favor, genera y publica una campaña completa para el siguiente caso:
    - Producto: {producto}
    - Público Objetivo: {publico}

    Sigue estos pasos OBLIGATORIAMENTE:
    1. Genera un hilo de TRES tweets creativos y atractivos. Llama a la herramienta `subir_tweet` UNA VEZ POR CADA TWEET. Deben ser tres llamadas separadas.
    2. Genera UN post profesional y bien estructurado para LinkedIn. Llama a la herramienta `subir_post_linkedin` para publicarlo.
    3. Genera UNA descripción llamativa para Instagram, incluyendo emojis relevantes y al menos 3 hashtags. Llama a la herramienta `subir_publicacion_instagram` para publicarla.
    4. Al finalizar todas las publicaciones, responde con un resumen de lo que hiciste.
    """

    print("\n El agente está procesando tu solicitud y generando la campaña...")
    # Invocar al agente para que realice la tarea.
    response = await agent_executor.ainvoke({"input": user_input})

    print("\n ¡Campaña completada exitosamente! ")
    print(f"\nRespuesta final del agente: {response['output']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n Saliendo del programa.")