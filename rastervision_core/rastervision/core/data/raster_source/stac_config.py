from typing import TYPE_CHECKING

from rastervision.pipeline.config import (Config, Field, register_config)
from rastervision.pipeline.file_system.utils import file_to_json

if TYPE_CHECKING:
    from pystac import Item, ItemCollection


@register_config('stac_item')
class STACItemConfig(Config):
    """Specify a raster via a STAC Item."""

    uri: str = Field(..., description='URI to a JSON-serialized STAC Item.')
    assets: list[str] | None = Field(
        None,
        description=
        'Subset of assets to use. This should be a list of asset keys')

    def build(self) -> 'Item':
        from pystac import Item

        item = Item.from_dict(file_to_json(self.uri))
        if self.assets is not None:
            item = subset_assets(item, self.assets)
        return item


@register_config('stac_item_collection')
class STACItemCollectionConfig(Config):
    """Specify a raster via a STAC ItemCollection."""

    uri: str = Field(
        ..., description='URI to a JSON-serialized STAC ItemCollection.')
    assets: list[str] | None = Field(
        None,
        description=
        'Subset of assets to use. This should be a list of asset keys')

    def build(self) -> 'ItemCollection':
        from pystac import ItemCollection

        items = ItemCollection.from_dict(file_to_json(self.uri))
        if self.assets is not None:
            items = [subset_assets(item, self.assets) for item in items]
            items = ItemCollection(items)
        return items


def subset_assets(item: 'Item', assets: list[str]) -> 'Item':
    """Return a copy of the Item with assets subsetted."""
    item = item.clone()
    src_assets = item.assets
    item.assets = {k: src_assets[k] for k in assets}
    return item
