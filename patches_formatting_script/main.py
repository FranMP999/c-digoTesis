'''
El siguiente es el script utilizado para el procesamiento de las imágenes 
satelitales Sentinel S2 en conjunto con los polígonos de parcelas de cultivos,
para formar tensores (band, time, x, y) con la agregación de productos Sentinel 2
en parches de 256x256 y matrices (x,y) con las anotaciones de cultivos asociadas
a la ubicación geográfica.

Utiliza las funciones definidas en el archivo aux_functions.py

Loopea a través de los tiles encontrados en `s2_path` y para cada tile
genera todos los patches de 256x256 asociados, guardando la metadata actualzada
cada 5 patches generados.
'''
import os
from dotenv import load_dotenv
from pathlib import Path
import time
import numpy as np
import pandas as pd 

import rasterio
import geopandas as gpd


from aux_functions import (
    get_id,
    get_labels_in_tile,
    get_annotation_raster,
    create_patch_tensor_rasterio,
)

load_dotenv()
in_path = Path(os.getenv("INPUT_DATA_PATH")) # dirección del directorio con los datos a procesar.
s2_path = in_path / "productos"
labels_path = in_path / "gsa_2022_selectedtiles.gpkg"
assert labels_path.exists(), "No existe archivo con labels"

out_path = Path(os.getenv("OUTPUT_DATA_PATH")) # dirección del directorio donde se almacenarán los datos procesados siguiendo el formato de https://huggingface.co/datasets/IGNF/PASTIS-HD/tree/main.
metadata_path = out_path / "metadata.geojson" #dirección de la metadata producida en el procesamiento.
s2_out_path = out_path / "DATA_S2" #dirección de los tensores de imágenes 4D producidos.
annotations_out_path = out_path / "ANNOTATIONS" #dirección de las matrices 2D con los labels producidos.
for path in [s2_out_path, annotations_out_path]:
    if not path.exists(): os.makedirs(path)

class_mapping_path = Path("class_mapping.csv")
class_mapping = (
    pd.read_csv(class_mapping_path, index_col=0)
    .iloc[:,0]
    .to_dict()
)


start = time.time()
#Leer los tiles únicos en s2_path y loopear procedimiento para cada tile
unique_tiles = set([
    str(path).split("_")[-2][1:]
    for path in s2_path.glob( f"*")
    ])
print(f"Tiles encontrados:\n\t", *list(unique_tiles), sep=" ")

metadata_rows = []
count=0
for tile_name in unique_tiles:
    print(f"Formateando tile {tile_name}...")
    sentinel_crs = get_crs(s2_path.glob(f"*{tile_name}*"))

    #Parcelas en tile
    labels_gdf = get_labels_in_tile(
        labels_path=labels_path,
        tile_name=tile_name,
        class_mapping=class_mapping,
        crs=sentinel_crs,
    )

    #Crear tensor con bandas y tiempo, en el patch n-ésimo
    array_size = 10980
    patch_size = 256
    final_n = (array_size//patch_size + 1)**2
    for patch_n in range(0, final_n):
        id = get_id(tile_name, patch_n)
        print(f"\tFormateando patch {patch_n} (id={id})...")

        time_series_tensor, raster_data = create_patch_tensor_rasterio(
            products_paths=s2_path.glob(f"*{tile_name}*"),
            patch_n=patch_n
        )
        # Crear el raster de etiquetado
        annotation_raster = get_annotation_raster(
                raster_data, 
                labels_gdf
        )

        # Guardar resultados
        np.save(
            s2_out_path / f"S2_{"{:05}".format(id)}.npy",
           time_series_tensor 
        )
        np.save(
            annotations_out_path / f"ParcelIDs_{"{:05}".format(id)}.npy",
            annotation_raster.astype(np.int16)
        )

        # Guardar metadata cada 5 patches
        metadata_rows.append({
            "id": id,
            "tile_name": tile_name,
            "patch_n": patch_n,
            "parcel_cover": (annotation_raster > 0).sum() / annotation_raster.size,
            "dates_S2": {
                i: pd.to_datetime(date).strftime("%Y%m%d")
                for i, date in enumerate(raster_data.dates)
            },
            "geometry": raster_data.bounds,
        })
        count = (count + 1)%5
        if count==0 :
            print("Tiempo de ejecución acumulado: ",
                  round((time.time() - start), 2), "[m]")
            #Almacenar metadata
            metadata_gdf = (
                gpd.GeoDataFrame(metadata_rows, geometry="geometry", crs=sentinel_crs)
                .set_index("id")
            )
            if metadata_path.exists():
                old_metadata_gdf = (
                    gpd.read_file(metadata_path).set_crs(sentinel_crs, allow_override=True)
                    .astype({"id": int})
                    .set_index("id")
                )
                metadata_gdf = pd.concat([
                    metadata_gdf,
                    old_metadata_gdf,
                    ]).reset_index().drop_duplicates(subset="id").set_index("id")
            with open(metadata_path, "w") as text_file:
                text_file.write(metadata_gdf.to_json())

end = time.time()
print("The time of execution of above program is :",
      (end-start)/60 , "m")
