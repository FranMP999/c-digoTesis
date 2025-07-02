from pathlib import Path
import pandas as pd 
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box

class RasterData:
    '''
    Clase para pasar la data geográfica (y metadata) del patch de una función a otra.
    Esta en conjunto con el np.ndarray del patch constituyen toda la información
    extraída en su creación.
    '''
    def __init__(
            self, 
            raster_reader: rasterio.io.DatasetReader,
            window: rasterio.windows.Window
    ):
        self.bounds = box(*raster_reader.window_bounds(window))
        self.transform = raster_reader.window_transform(window)
        self.shape = (256, 256)
        
    def set_dates(self, dates):
        self.dates = dates
        return self

def get_band_paths(product_path: Path):
    '''
    Obtiene una lista con los paths de las bandas a partir del path de producto.
    '''
    return np.sort(list(product_path.glob(
        f"*_*_B*"
        )))

# Para acceder archivos de productos
def path2band(path: Path):
    '''
    Obtiene el código de banda de un producto Sentinel-2 a partir de su path.
    '''
    return str(path).split("/")[-1].split("_")[-2]

def path2date(path: Path):
    '''
    Entrega el datetime asociado a un producto Sentinel-2 a partir de su path.
    '''
    return pd.to_datetime(str(path).split("/")[-1].split("_")[1][:8])

def get_crs(products_paths):
    '''
    Loopea en todos los productos buscando crs.
    Se asegura de que el crs exista y sea consistente.

    Según lo explorado no todas los productos tienen crs, pero basta con que 
    alguno lo tenga y que los que tengan, tengan el mismo.
    '''
    crs_arr = []
    for product_path in products_paths:
        for band_path in get_band_paths(product_path):
            crs = rasterio.open(band_path).crs.to_epsg()
            if crs is not None:
                crs_arr.append(crs)
    crs_arr = np.array(crs_arr)
    assert len(crs_arr) > 0, "No se encontró crs."
    return f"EPSG:{crs_arr[0]}"


# Para trabajar patches
def patch_coors(n: int, patch_size=256, array_size=1830):
    '''
    Retorna las coordenadas del punto superior izquierdo del patch n-ésimo.
    Supone que todos los productos son de igual tamaño.
    '''
    lim = patch_size *(array_size // patch_size +1 )
    x = (n * patch_size) % lim
    if ((n+1) * patch_size) % lim == 0:# Condición último patch
        x = array_size - patch_size

    y= ((n * patch_size) // lim) * patch_size
    if ((n * patch_size) // lim) * patch_size + patch_size > array_size:# Condición último patch
        y = array_size - patch_size
    return (x,y)



def get_labels_in_tile(labels_path: Path, tile_name: str, class_mapping: dict, crs) -> gpd.GeoDataFrame:
    '''
    Retorna un geopandas dataframe con las parcelas en el tile
    a partir de un path, el nombre del tile, el mapeo hcat4-clases y un crs.
    '''
    return (
        gpd.read_file(
            labels_path,
            where=f"name='{tile_name}'",
        )
        .to_crs(crs)
        .assign(polygon=lambda df: df.geometry.map(lambda x: x.geoms[0]))
        .assign(crop_class=lambda df: df.hcat4_code.map(class_mapping))
    )


def get_patch_rasterio(
        raster_reader:rasterio.io.DatasetReader,
        n: int, patch_size=256, get_data=False
)-> np.ndarray | RasterData :
    '''
    Retorna el patch n-ésimo a partir de un Dataset Reader de rasterio
    '''
    array_size = raster_reader.shape[0]

    x_patch, y_patch = patch_coors(n, patch_size, array_size)
    window = rasterio.windows.Window.from_slices(
            cols=slice(x_patch, x_patch + 256),
            rows=slice(y_patch, y_patch + 256),
        )
    if get_data:
        return raster_reader.read( 1, window=window), RasterData(raster_reader, window)
    else:
        return raster_reader.read( 1, window=window)

def create_patch_tensor_rasterio(products_paths: Path, patch_n: int) -> np.ndarray | RasterData :
    '''
    Retorna tupla (tensor, RasterData) con tensor del parche completo.
    Recibe un iterable con los paths de todos los productos del tile y el número de patch deseado.
    '''
    frames = []
    #Se ordenan los path de productos por fecha
    dates, sorted_paths = (lambda series: (series.index, series.values))(
            pd.Series({ path2date(get_band_paths(path)[0]): path for path in products_paths })
            .sort_index()
            )

    for product_path in sorted_paths:
        band_arrays = []
        i=0
        for band_path in get_band_paths(product_path):
            with rasterio.open(band_path) as src:
                if i==0:
                    band_raster, patch_data = get_patch_rasterio(src, patch_n, get_data=True)
                    i+=1
                else:
                    band_raster = get_patch_rasterio(src, patch_n)
            band_arrays.append(band_raster)
        if len(band_arrays) == 12:
            stack = np.stack(band_arrays, axis=0)  # (bands, N, N)
            frames.append(stack)
        else: print(f"PRODUCTO NO TIENE LAS 12 BANDAS: {product_path} ")
    tensor_final = np.stack(frames, axis=0)
    patch_data.set_dates(dates)
    return tensor_final, patch_data  # (temporal, bands, N, N)


def get_annotation_raster(patch_data: RasterData, labels_gdf: gpd.GeoDataFrame) -> np.ndarray:
        tensor_bounds = patch_data.bounds
        sel_parcels = labels_gdf[ labels_gdf.intersects(tensor_bounds) ]
        shapes = list(zip(sel_parcels.polygon, sel_parcels.crop_class))
        return rasterio.features.rasterize(
            shapes,
            out_shape = patch_data.shape,
            fill = 0,
            transform = patch_data.transform,
            all_touched = False, # Esto lo tengo que revisar bien
            dtype = None
        )


def get_id(tile_name: str, patch_n: int):
    array_size = 10980
    tiles = [
        "31TBF",
        "29TNF",
        "30UXU",
        "32TPP",
        "32UMC",
        "29UNU",
        "33TVN",
        "31UFU",
        "35TMH",
        "32VNH",
    ]
    tile_map = {tile: i for i, tile in enumerate(tiles)}
    patchesxtile = (array_size//256 + 1)**2
    total_n_patches = patchesxtile * len(tiles)
    id = (tile_map[tile_name] * patchesxtile) + patch_n

    assert patch_n < patchesxtile, "número de patch no válido"
    assert tile_name in tiles, "tile no válido"
    return id


def which_patch(id:str):
    '''
    Recibe un id y entrega a qué tile y patch corresponde.
    No se utiliza en el código pero puede ser útil.
    '''
    array_size = 10980
    tiles = [
        "31TBF",
        "29TNF",
        "30UXU",
        "32TPP",
        "32UMC",
        "29UNU",
        "33TVN",
        "31UFU",
        "35TMH",
        "32VNH",
    ]
    patchesxtile = (array_size//256 + 1)**2

    tile_name = tiles[id//patchesxtile]
    patch_n = id%patchesxtile
    print(f"El id {id} está asocidado al patch {patch_n} del tile {tile_name}.")
    return tile_name, patch_n

### LEGACY CODE ###

def get_patch_xarray(xarray, n, patch_size=256):
    '''
    CUIDADO CON ESTA FUNCIÓN EL COMPORTAMIENTO DE ISEL NO ES EL ESPERADO!!!!!!!

    xarray.isel( x=slice(x_patch, x_patch + 256), y=slice(y_patch, y_patch + 256)) 
    !=  xarray.isel(band=0).to_array()[slice(x_patch, x_patch + 256), (y_patch, y_patch + 256)]


    Retorna el patch n-ésimo.
    Supone que todos los productos son de igual tamaño.
    '''
    array_size = xarray.x.size

    patchesxtile = (array_size//patch_size + 1)**2
    assert n < patchesxtile, "número de patch no válido"

    x_patch, y_patch = patch_coors(n, patch_size, array_size)
    return xarray.isel(
            x=slice(x_patch, x_patch + 256),
            y=slice(y_patch, y_patch + 256),
        )

def create_patch_tensor_xarray(products_paths, patch_n):
        multiband_tensors = []
        for product_path in products_paths:
            multiband_tensors.append(
                xr.concat([
                    get_patch(
                        xr.open_dataset(band_path, engine="rasterio", band_as_variable=True)
                        #.drop_dims("band")
                        .assign(band=path2band(band_path)).set_coords("band")
                        .assign(time=path2date(band_path)).set_coords("time"),
                        n = patch_n
                    )
                    for band_path in get_band_paths(product_path)
                ], dim="band")
            )
        return xr.concat(multiband_tensors, dim="time")
