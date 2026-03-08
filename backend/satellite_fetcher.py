import os
from sentinelhub import SHConfig, SentinelHubRequest, MimeType, CRS, BBox, DataCollection, bbox_to_dimensions

# LOAD CONFIG FROM settings.py
from config.settings import SH_CLIENT_ID, SH_CLIENT_SECRET

def download_satellite_image(lat, lon, size=1024, save_path="data/sentinel_images/image.tif"):
    config = SHConfig()
    config.sh_client_id = SH_CLIENT_ID
    config.sh_client_secret = SH_CLIENT_SECRET

    # Bounding box around location
    bbox = BBox(bbox=[lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01], crs=CRS.WGS84)
    resolution = bbox_to_dimensions(bbox, resolution=10)

    request = SentinelHubRequest(
        data_folder="data/sentinel_images",
        evalscript="""
            // Returns RGB true color
            function setup() {
                return { input: ["B04", "B03", "B02"], output: { bands: 3 } };
            }
            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
        """,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=("2020-06-01", "2020-06-30")
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=resolution,
        config=config
    )

    image = request.get_data(save_data=True)
    print(f"Satellite image saved → {save_path}")


if __name__ == "__main__":
    lat = 12.9716   # Example: Bengaluru latitude
    lon = 77.5946   # Example: Bengaluru longitude
    download_satellite_image(lat, lon)
