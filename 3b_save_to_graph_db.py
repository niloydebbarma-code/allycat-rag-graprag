import asyncio
import json
import logging
import os
import time
import sys
from typing import Any, Dict, List, Optional
from my_config import MY_CONFIG
from neo4j import GraphDatabase, Driver
from tqdm import tqdm
from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_DATA_DIR = MY_CONFIG.GRAPH_DATA_DIR
GRAPH_DATA_FILE = os.path.join(GRAPH_DATA_DIR, "graph_data.json")

class Neo4jConnection:
    def __init__(self):
        self.uri = MY_CONFIG.NEO4J_URI
        self.username = MY_CONFIG.NEO4J_USER
        self.password = MY_CONFIG.NEO4J_PASSWORD
        self.database = getattr(MY_CONFIG, "NEO4J_DATABASE", None)
        if not self.uri:
            raise ValueError("NEO4J_URI config is required")
        if not self.username:
            raise ValueError("NEO4J_USERNAME config is required")
        if not self.password:
            raise ValueError("NEO4J_PASSWORD config is required")
        if not self.database:
            raise ValueError("NEO4J_DATABASE config is required")
        self.driver: Optional[Driver] = None
    
    async def connect(self):
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
                
                # Verify connectivity
                await asyncio.get_event_loop().run_in_executor(
                    None, self.driver.verify_connectivity
                )
                logger.info(f"Connected to Neo4j at {self.uri}")
                
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self.driver = None
    
    async def disconnect(self):
        if self.driver:
            await asyncio.get_event_loop().run_in_executor(
                None, self.driver.close
            )
            self.driver = None
            logger.info("Disconnected from Neo4j")
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        if not self.driver:
            raise ConnectionError("Not connected to Neo4j database")
        
        def run_query():
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]
                summary = result.consume()
                return records, summary
        
        return await asyncio.get_event_loop().run_in_executor(None, run_query)

neo4j_connection = Neo4jConnection()

app = FastMCP("Neo4j Graph Data Upload Server")

@app.tool()
async def execute_cypher(query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database",
                    "details": "Check connection settings and network connectivity"
                }
        
        records, summary = await neo4j_connection.execute_query(query, parameters)
        
        return {
            "status": "success",
            "query": query,
            "parameters": parameters or {},
            "records": records,
            "record_count": len(records),
            "execution_time_ms": summary.result_available_after,
            "summary": {
                "query_type": summary.query_type,
                "counters": dict(summary.counters) if summary.counters else {}
            }
        }
        
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        return {
            "status": "error",
            "query": query,
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.tool()
async def get_database_schema() -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database"
                }
        
        labels_records, _ = await neo4j_connection.execute_query("CALL db.labels()")
        labels = [record["label"] for record in labels_records]
        
        rel_records, _ = await neo4j_connection.execute_query("CALL db.relationshipTypes()")
        relationships = [record["relationshipType"] for record in rel_records]
        
        prop_records, _ = await neo4j_connection.execute_query("CALL db.propertyKeys()")
        properties = [record["propertyKey"] for record in prop_records]

        try:
            constraint_records, _ = await neo4j_connection.execute_query("SHOW CONSTRAINTS")
            constraints = [dict(record) for record in constraint_records]
        except Exception:
            constraints = []
        
        try:
            index_records, _ = await neo4j_connection.execute_query("SHOW INDEXES")
            indexes = [dict(record) for record in index_records]
        except Exception:
            indexes = []
        
        return {
            "status": "success",
            "schema": {
                "node_labels": labels,
                "relationship_types": relationships,
                "property_keys": properties,
                "constraints": constraints,
                "indexes": indexes
            }
        }
        
    except Exception as e:
        logger.error(f"Schema retrieval failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.tool()
async def get_node_count(label: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database"
                }
        
        if label:
            query = f"MATCH (n:`{label}`) RETURN count(n) as count"
        else:
            query = "MATCH (n) RETURN count(n) as count"
        
        records, _ = await neo4j_connection.execute_query(query)
        count = records[0]["count"] if records else 0
        
        return {
            "status": "success",
            "label": label,
            "count": count
        }
        
    except Exception as e:
        logger.error(f"Node count failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.tool()
async def get_relationship_count(relationship_type: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database"
                }
        
        if relationship_type:
            query = f"MATCH ()-[r:`{relationship_type}`]-() RETURN count(r) as count"
        else:
            query = "MATCH ()-[r]-() RETURN count(r) as count"
        
        records, _ = await neo4j_connection.execute_query(query)
        count = records[0]["count"] if records else 0
        
        return {
            "status": "success",
            "relationship_type": relationship_type,
            "count": count
        }
        
    except Exception as e:
        logger.error(f"Relationship count failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.tool()
async def health_check() -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
        
        if not neo4j_connection.driver:
            return {
                "status": "unhealthy",
                "reason": "Unable to connect to Neo4j database",
                "configuration": {
                    "uri": neo4j_connection.uri,
                    "database": neo4j_connection.database,
                    "username": neo4j_connection.username
                }
            }
        
        # A simple query to test connectivity
        records, _ = await neo4j_connection.execute_query("RETURN 1 as test")
        
        if records and records[0]["test"] == 1:
            return {
                "status": "healthy",
                "database": neo4j_connection.database,
                "uri": neo4j_connection.uri,
                "ssl_enabled": neo4j_connection.uri.startswith(('neo4j+s://', 'bolt+s://')),
                "message": "Neo4j connection is working properly"
            }
        else:
            return {
                "status": "unhealthy",
                "reason": "Query execution failed or returned unexpected results"
            }
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "reason": str(e),
            "error_type": type(e).__name__
        }


async def clear_database_impl() -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database"
                }
        
        print("Clearing database...")
        
        # Get current counts before clearing
        node_count_query = "MATCH (n) RETURN count(n) as count"
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
        
        node_records, _ = await neo4j_connection.execute_query(node_count_query)
        rel_records, _ = await neo4j_connection.execute_query(rel_count_query)
        
        nodes_before = node_records[0]["count"] if node_records else 0
        rels_before = rel_records[0]["count"] if rel_records else 0
        
        # Clear all relationships first
        clear_rels_query = "MATCH ()-[r]->() DELETE r"
        await neo4j_connection.execute_query(clear_rels_query)
        
        # Clear all nodes
        clear_nodes_query = "MATCH (n) DELETE n"
        await neo4j_connection.execute_query(clear_nodes_query)
        
        print(f"✅ Database cleared: {nodes_before} nodes, {rels_before} relationships removed")
        
        return {
            "status": "success",
            "message": "Database cleared successfully",
            "statistics": {
                "nodes_removed": nodes_before,
                "relationships_removed": rels_before
            }
        }
        
    except Exception as e:
        logger.error(f"Database clear failed: {str(e)}")
        print(f"❌ Database clear failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.tool()
async def clear_database() -> Dict[str, Any]:
    return await clear_database_impl()


async def upload_graph_data_impl() -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database"
                }
        
        # Step 1: Clear existing data
        print("Clearing existing database...")
        clear_result = await clear_database_impl()
        if clear_result["status"] != "success":
            return clear_result
        
        # Check if graph data file exists
        if not os.path.exists(GRAPH_DATA_FILE):
            return {
                "status": "error",
                "error": f"Graph data file not found: {GRAPH_DATA_FILE}"
            }
        
        # Load graph data
        with open(GRAPH_DATA_FILE, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        if not isinstance(graph_data, dict) or 'nodes' not in graph_data:
            return {
                "status": "error",
                "error": "Invalid graph data format. Expected JSON with 'nodes' array"
            }
        
        nodes = graph_data.get('nodes', [])
        # Handle both 'relationships' and 'edges' keys for compatibility
        relationships = graph_data.get('relationships', graph_data.get('edges', []))
        
        total_items = len(nodes) + len(relationships)
        print(f"Processing {len(nodes)} nodes and {len(relationships)} relationships...")
        
        upload_stats = {
            "nodes_processed": 0,
            "nodes_created": 0,
            "relationships_processed": 0,
            "relationships_created": 0,
            "errors": []
        }
        
        # Progress bar for nodes
        print("Creating nodes...")
        with tqdm(total=len(nodes), desc="Nodes", unit="node", ncols=80, leave=False) as pbar:
            for node in nodes:
                try:
                    upload_stats["nodes_processed"] += 1
                    
                    node_id = node.get('id')
                    labels = node.get('labels', [])
                    properties = node.get('properties', {})
                    
                    if not node_id or not labels:
                        upload_stats["errors"].append(f"Node missing id or labels: {node}")
                        pbar.update(1)
                        continue
                    
                    # Create node with labels
                    labels_str = ':'.join([f"`{label}`" for label in labels])
                    query = f"MERGE (n:{labels_str} {{id: $id}}) SET n += $props RETURN n"
                    
                    await neo4j_connection.execute_query(query, {
                        "id": node_id,
                        "props": properties
                    })
                    
                    upload_stats["nodes_created"] += 1
                    pbar.update(1)
                    
                except Exception as e:
                    upload_stats["errors"].append(f"Node upload error: {str(e)}")
                    pbar.update(1)
        
        # Progress bar for relationships  
        print(" Creating relationships...")
        with tqdm(total=len(relationships), desc="Relationships", unit="rel", ncols=80, leave=False) as pbar:
            for rel in relationships:
                try:
                    upload_stats["relationships_processed"] += 1

                    source_id = rel.get('startNode')
                    target_id = rel.get('endNode') 
                    rel_type = rel.get('type')
                    properties = {}
                    for key, value in rel.items():
                        if key not in ['startNode', 'endNode', 'type']:
                            properties[key] = value
                    
                    if not source_id or not target_id or not rel_type:
                        upload_stats["errors"].append(f"Relationship missing startNode, endNode, or type: {rel}")
                        pbar.update(1)
                        continue
                    
                    # Create relationship
                    query = f"""
                    MATCH (a {{id: $source_id}})
                    MATCH (b {{id: $target_id}})
                    MERGE (a)-[r:`{rel_type}`]->(b)
                    SET r += $props
                    RETURN r
                    """
                    
                    await neo4j_connection.execute_query(query, {
                        "source_id": source_id,
                        "target_id": target_id,
                        "props": properties
                    })
                    
                    upload_stats["relationships_created"] += 1
                    pbar.update(1)
                    
                except Exception as e:
                    upload_stats["errors"].append(f"Relationship upload error: {str(e)}")
                    pbar.update(1)
        
        # Calculate success percentage
        nodes_success_rate = (upload_stats["nodes_created"] / len(nodes) * 100) if nodes else 100
        rels_success_rate = (upload_stats["relationships_created"] / len(relationships) * 100) if relationships else 100
        overall_success_rate = (upload_stats["nodes_created"] + upload_stats["relationships_created"]) / total_items * 100 if total_items else 100
        
        result = {
            "status": "success",
            "message": "Graph data upload completed",
            "statistics": upload_stats,
            "success_rates": {
                "nodes": f"{nodes_success_rate:.1f}%",
                "relationships": f"{rels_success_rate:.1f}%",
                "overall": f"{overall_success_rate:.1f}%"
            },
            "source_file": GRAPH_DATA_FILE
        }
        
        # Print upload success summary
        print("\n✅ Graph data upload completed!")
        print(f"Nodes: {upload_stats['nodes_created']}/{len(nodes)} ({nodes_success_rate:.1f}%)")
        print(f" Relationships: {upload_stats['relationships_created']}/{len(relationships)} ({rels_success_rate:.1f}%)")
        print(f"Overall success: {overall_success_rate:.1f}%")
        if upload_stats['errors']:
            print(f"⚠️ rrors: {len(upload_stats['errors'])}")
        
        return result
        
    except Exception as e:
        logger.error(f"Graph data upload failed: {str(e)}")
        print(f"❌ Graph data upload failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.tool()
async def upload_graph_data() -> Dict[str, Any]:
    return await upload_graph_data_impl()


@app.tool()
async def check_graph_data_file() -> Dict[str, Any]:
    try:
        if not os.path.exists(GRAPH_DATA_FILE):
            return {
                "status": "not_found",
                "path": GRAPH_DATA_FILE,
                "message": "Graph data file does not exist"
            }
        
        # Get file stats
        file_stats = os.stat(GRAPH_DATA_FILE)
        file_size = file_stats.st_size
        
        # Try to parse the JSON to validate format
        try:
            with open(GRAPH_DATA_FILE, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            nodes_count = len(graph_data.get('nodes', []))
            relationships_count = len(graph_data.get('relationships', []))
            
            return {
                "status": "found",
                "path": GRAPH_DATA_FILE,
                "file_size_bytes": file_size,
                "nodes_count": nodes_count,
                "relationships_count": relationships_count,
                "valid_json": True
            }
            
        except json.JSONDecodeError as e:
            return {
                "status": "invalid",
                "path": GRAPH_DATA_FILE,
                "file_size_bytes": file_size,
                "valid_json": False,
                "json_error": str(e)
            }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.tool()
async def get_connection_info() -> Dict[str, Any]:
    try:
        # Always return configuration info even if not connected
        deployment_type = "Self-hosted"
        if "databases.neo4j.io" in neo4j_connection.uri:
            deployment_type = "Neo4j Aura"
        elif "sandbox" in neo4j_connection.uri:
            deployment_type = "Neo4j Sandbox"
        elif any(cloud in neo4j_connection.uri for cloud in ["aws", "gcp", "azure"]):
            deployment_type = "Enterprise Cloud"
        
        connection_info = {
            "status": "success",
            "connection": {
                "uri": neo4j_connection.uri,
                "database": neo4j_connection.database,
                "username": neo4j_connection.username,
                "deployment_type": deployment_type,
                "ssl_enabled": neo4j_connection.uri.startswith(('neo4j+s://', 'bolt+s://')),
                "connected": neo4j_connection.driver is not None
            },
            "capabilities": {
                "cypher_queries": True,
                "schema_inspection": True,
                "bulk_operations": True,
                "graph_algorithms": "unknown",
                "multi_database": "unknown"
            }
        }
        
        if neo4j_connection.driver:
            try:
                server_info_records, _ = await neo4j_connection.execute_query(
                    "CALL dbms.components() YIELD name, versions, edition"
                )
                connection_info["server_info"] = server_info_records[0] if server_info_records else {}
            except Exception:
                connection_info["server_info"] = {}
        
        return connection_info
        
    except Exception as e:
        logger.error(f"Connection info retrieval failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    import sys
    try:
        asyncio.run(neo4j_connection.connect())
        print(f"Looking for graph data at: {GRAPH_DATA_FILE}")
        print(f"File exists: {os.path.exists(GRAPH_DATA_FILE)}")
        
        result = asyncio.run(upload_graph_data_impl())
        print(f"Upload result: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'error':
            print(f"❌ Error details: {result.get('error', 'Unknown error')}")
            if 'error_type' in result:
                print(f"Error type: {result['error_type']}")
        elif result.get('status') == 'success':
            stats = result.get('statistics', {})
            print(" Upload statistics:")
            print(f"   Nodes processed: {stats.get('nodes_processed', 0)}")
            print(f"   Nodes created: {stats.get('nodes_created', 0)}")
            print(f"   Relationships processed: {stats.get('relationships_processed', 0)}")
            print(f"   Relationships created: {stats.get('relationships_created', 0)}")
            if stats.get('errors'):
                print(f"   Errors encountered: {len(stats['errors'])}")
        
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.warning(f"Connection Warning: {e}")