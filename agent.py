
# Implementación del agente de IA con ciclo de tool calling manual, ya que la pasé muy mal haciendo con langchain
import os
import asyncio
import json
from dotenv import load_dotenv

# Importaciones esenciales para el cliente MCP y el modelo
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
# Necesitamos el objeto Message para el formato de chat
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

def create_system_prompt():
    """Define el prompt de sistema para el Community Manager."""
    return (
        "Eres un experto Community Manager. Tu tarea es generar y publicar una campaña completa "
        "en redes sociales usando las herramientas disponibles. Debes seguir las instrucciones "
        "del usuario al pie de la letra, usando las herramientas `subir_tweet`, `subir_post_linkedin` "
        "y `subir_publicacion_instagram`."
    )

def create_user_input(producto, publico):
    """Genera el input detallado para el modelo."""
    return f"""
    Genera y publica una campaña completa para el siguiente caso:
    - Producto: {producto}
    - Público Objetivo: {publico}

    Sigue estos pasos OBLIGATORIAMENTE:
    1. Genera un hilo de TRES tweets creativos y atractivos. Llama a la herramienta `subir_tweet` UNA VEZ POR CADA TWEET. Deben ser tres llamadas separadas.
    2. Genera UN post profesional y bien estructurado para LinkedIn. Llama a la herramienta `subir_post_linkedin` para publicarlo.
    3. Genera UNA descripción llamativa para Instagram, incluyendo emojis relevantes y al menos 3 hashtags. Llama a la herramienta `subir_publicacion_instagram` para publicarla.
    4. Al finalizar todas las publicaciones, responde con un resumen de lo que hiciste.
    """

# Función Principal del Agente 

async def run_campaign_agent():
    # Configuración Inicial
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("No se encontró la GOOGLE_API_KEY. Crea un archivo .env con tu clave.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)
    client = MultiServerMCPClient({
        "campaign_tools": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",
        }
    })

    print(" Conectando al servidor MCP para obtener las herramientas disponibles...")
    mcp_tools = await client.get_tools()
    
    tools_for_llm = mcp_tools 
    
    tool_names = {tool.name: tool for tool in mcp_tools}
    print(f"  Herramientas obtenidas: {[t.name for t in mcp_tools]}")

    # Interacción con el Usuario
    print("\n Asistente de Campañas para Redes Sociales Activado ")
    print("-------------------------------------------------------")
    producto = input("Por favor, ingresa el nombre del producto: ")
    publico = input("Ahora, describe el público objetivo: ")

    # Iniciar la Conversación (El Agente es una secuencia de mensajes)
    system_prompt = create_system_prompt()
    user_input = create_user_input(producto, publico)
    
    # Lista de mensajes para la conversación: incluye el System Prompt inicial
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    print("\n El agente está procesando tu solicitud y generando la campaña...")

    # El Bucle de Tool Calling Manual
    while True:
        # Invocar al modelo
        # Pasamos el historial de mensajes y las herramientas disponibles
        response = await llm.ainvoke(messages, tools=tools_for_llm)
        messages.append(response) # Añadir la respuesta del modelo al historial
        
        # Comprobar si el modelo quiere llamar a una herramienta
        if response.tool_calls:
            print("\n Modelo Solicitó Tool Calls. Ejecutando...")
            
            tool_response_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                print(f"  - Ejecutando: {tool_name} con argumentos: {tool_args}")

                # Obtenemos el objeto StructuredTool a partir del nombre
                tool_object = tool_names[tool_name]
                
                # Invocamos la herramienta usando el método ainvoke de LangChain
                # Le pasamos los argumentos (tool_args) como un diccionario.
                tool_output = await tool_object.ainvoke(tool_args)
                # -----------------------------------
                
                print(f"  - Salida de la Tool: {tool_output[:50]}...")  

                # Crear el mensaje de respuesta de la herramienta para enviárselo al modelo
                tool_response_messages.append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_call_id,
                    )
                )
            
            # Añadir la salida de las herramientas al historial y volver a llamar al modelo
            messages.extend(tool_response_messages)
        else:
            # Si no hay tool_calls, es la respuesta final de texto
            print("\n ¡Campaña completada exitosamente! ")
            print("-------------------------------------------------------")
            print(f"\nRespuesta final del agente:\n{response.content}")
            break

if __name__ == "__main__":
    try:
        asyncio.run(run_campaign_agent())
    except KeyboardInterrupt:
        print("\n Saliendo del programa.")
    except Exception as e:
        print(f"\nOcurrió un error crítico: {e}")