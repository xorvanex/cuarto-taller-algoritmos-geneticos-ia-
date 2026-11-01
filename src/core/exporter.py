"""
Módulo de exportación desacoplada de datasets analíticos (JSON y JS).
====================================================================
Garantiza el desacoplamiento entre el backend de simulación Python
y la interfaz gráfica web Glassmorphism / Dark-Tech.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple


def export_dataset(
    data: Dict[str, Any],
    var_name: str,
    file_stem: str,
    project_root: str
) -> Tuple[str, str]:
    """
    Exporta simultáneamente el dataset en dos destinos:
      1. data/{file_stem}.json : Archivo estándar para análisis offline y persistencia.
      2. src/web/data/{file_stem}.js : Inyección en window.{var_name} para la web sin CORS.
    """
    data_dir = os.path.join(project_root, "data")
    web_data_dir = os.path.join(project_root, "src", "web", "data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(web_data_dir, exist_ok=True)

    json_path = os.path.join(data_dir, f"{file_stem}.json")
    js_path = os.path.join(web_data_dir, f"{file_stem}.js")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    js_content = (
        f"/** Dataset analitico oficial - Problema: {file_stem} */\n"
        f"window.{var_name} = {json.dumps(data, ensure_ascii=False)};\n"
    )
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_content)

    print(f"\n[EXPORTACION DE RESULTADOS: {file_stem.upper()}]")
    print(f"  [OK] Dataset JSON: {json_path}")
    print(f"  [OK] Dataset JS:   {js_path}")

    return json_path, js_path
