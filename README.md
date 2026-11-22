# TP3 Regularización en 20 Newsgroups

El [notebook del TP3](TP3_Regularizacion_TNG.ipynb) contiene la comparación controlada, la búsqueda de hiperparámetros, la confirmación y la evaluación final. Los diagnósticos complementarios se evaluaron en validación. Los resultados y las siete figuras están guardados en el notebook para su lectura directa.

## Reconstruir las salidas en Jupyter

Desde esta carpeta, prepará un entorno con Python 3.12 e instalá las dependencias. En Windows PowerShell podés usar estos comandos.

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m jupyterlab TP3_Regularizacion_TNG.ipynb

Seleccioná el kernel Python 3 de ese entorno y usá **Reiniciar el kernel y ejecutar todas las celdas**. Las celdas reconstruyen tablas y figuras leyendo los registros incluidos. La reconstrucción funciona con la carpeta de entrega y el entorno de lectura instalado, sin descargar 20 Newsgroups, entrenar modelos ni evaluar test.

## Entorno experimental

Las corridas se hicieron en Windows x64 con Python 3.12.5 y PyTorch 2.14.0+cpu. Se usaron NumPy 2.5.3, scikit-learn 1.9.1 y NLTK 3.10.3. La búsqueda usó Optuna 5.0.0. Las dependencias fijadas en `requirements.txt` corresponden a la lectura y reconstrucción de las salidas.

## Preparación de la caché de datos

El TP3 reutilizó la caché del TP2. Para prepararla, se cargó 20 Newsgroups con `fetch_20newsgroups`, apuntando `data_home` a `<directorio>/sklearn` y con la descarga habilitada. El archivo `sklearn/20news-bydate_py3.pkz` contiene los subconjuntos oficiales de entrenamiento y test. Con `nltk.download` se guardaron los recursos `stopwords`, `wordnet`, `omw-1.4`, `punkt` y `punkt_tab` en `<directorio>/nltk_data`.

Los scripts experimentales recibieron esa raíz por `--data-dir` y leyeron 20 Newsgroups con `download_if_missing=False`.

## Orden de ejecución experimental

1. [`piloto.py`](codigo/piloto.py) corrió las seis configuraciones de referencia. `informe_piloto.py` resumió las curvas y los tiempos.
2. [`busqueda_optuna.py`](codigo/busqueda_optuna.py) exploró los hiperparámetros por familia. `informe_optuna.py` verificó los ensayos y la promoción de candidatos.
3. [`confirmacion.py`](codigo/confirmacion.py) comparó los candidatos con las anclas del piloto y fijó la selección por validación. `informe_confirmacion.py` verificó la selección y los checkpoints.
4. [`test_final.py`](codigo/test_final.py) evaluó los 18 checkpoints fijados en validación.
5. [`diagnosticos.py`](codigo/diagnosticos.py) corrió los ensayos complementarios, evaluados solo en validación. `informe_diagnosticos.py` resumió sus resultados.

La carpeta de entrega incluye el código experimental para consulta y los registros para reconstruir las salidas. La ejecución original también utilizó la caché, los scripts de informe y archivos intermedios, incluidos los checkpoints, que se conservaron en el espacio de trabajo.

## Archivos de la entrega

- [Notebook](TP3_Regularizacion_TNG.ipynb) con metodología, análisis y salidas.
- [Registros y figuras](resultados/) con las métricas del piloto, Optuna, confirmación, test y diagnósticos.
- [Código experimental](codigo/) con las implementaciones usadas en las corridas, para consultar el procedimiento.
- [Dependencias](requirements.txt) para reconstruir las salidas del notebook.
