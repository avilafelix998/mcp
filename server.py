# Este archivo define el servidor MCP (Model Context Protocol).
# Su función es exponer un conjunto de "herramientas" (tools) que un agente de IA

from mcp.server.fastmcp import FastMCP

# Creamos una instancia del servidor MCP
mcp = FastMCP("ServidorDeCampanasSocialMedia")

# Definimos las herramientas que el servidor ofrecerá.
# Usamos el decorador @mcp.tool para exponer cada función como una herramienta disponible.

@mcp.tool("subir_tweet")
def subir_tweet(contenido: str):
    """
    Publica un único tweet en la plataforma X (anteriormente Twitter).
    Debe ser llamado por cada tweet individualmente en un hilo.

    Args:
        contenido (str): El texto del tweet a publicar. Debe ser conciso y directo.
    """
    print("--- EJECUTANDO HERRAMIENTA: subir_tweet ---")
    print(f"Contenido del Tweet: {contenido}")
    print("--- TWEET PUBLICADO (SIMULADO) ---\n")
    return f"El tweet que comienza con '{contenido[:30]}...' fue publicado exitosamente."

@mcp.tool("subir_post_linkedin")
def subir_post_linkedin(contenido: str):
    """
    Publica un post de formato profesional en la plataforma LinkedIn.

    Args:
        contenido (str): El texto del post para LinkedIn. Debe tener un tono profesional y estar bien estructurado.
    """
    print("--- EJECUTANDO HERRAMIENTA: subir_post_linkedin ---")
    print(f"Contenido de LinkedIn:\n{contenido}")
    print("--- POST DE LINKEDIN PUBLICADO (SIMULADO) ---\n")
    return "Post de LinkedIn publicado exitosamente."

@mcp.tool("subir_publicacion_instagram")
def subir_publicacion_instagram(contenido: str):
    """
    Publica una descripción para una foto o video en Instagram.

    Args:
        contenido (str): La descripción para la publicación de Instagram. Debe ser atractiva, usar emojis y hashtags relevantes.
    """
    print("--- EJECUTANDO HERRAMIENTA: subir_publicacion_instagram ---")
    print(f"Contenido de Instagram:\n{contenido}")
    print("--- PUBLICACIÓN DE INSTAGRAM REALIZADA (SIMULADO) ---\n")
    return "Publicación de Instagram realizada exitosamente."

# Ponemos el servidor a escuchar peticiones.
# Usamos el transporte "streamable-http" que es el estándar para comunicación por red.
if __name__ == "__main__":
    print(" Iniciando Servidor de Herramientas MCP en http://localhost:8000/mcp")
    print("   El servidor está listo para recibir peticiones del agente.")
    print("   Presiona CTRL+C para detener el servidor.")
    mcp.run(transport="streamable-http")