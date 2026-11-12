import os
import sys

import config

sys.path.insert(0, config.TOOL_DIR)
from estimate_index_storage import estimate_index_storage_detail


def estimate_index_storage(label, props, driver=None, on="node"):
    kind = "relationship" if on == "relationship" else "node"
    if driver is None:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH)
        try:
            return estimate_index_storage_detail(label, list(props), driver, kind)["mb"]
        finally:
            driver.close()
    return estimate_index_storage_detail(label, list(props), driver, kind)["mb"]
