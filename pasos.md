 primero se clona el repo

 luego instalas de forma global el `uv`

creas un entorno con `uv venv venv`

activas el etorno `venv\Scripts\activate`

instalas todas las dependecias usando el requirements con el siguiente comando `uv pip install -r requirements.txt`

para hacer correr los archivos lo haces por separado

en la termianl 1 luego de activar tu entorno pones `uv run python server.py`

en la termianl 2 luego de activar tu entorno pones `uv run python agent.py` 
