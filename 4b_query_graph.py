import os
import time
import logging
import json
from typing import List, Dict, Optional, Any
from my_config import MY_CONFIG
from neo4j import GraphDatabase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.litellm import LiteLLM
import query_utils

os.environ['HF_ENDPOINT'] = MY_CONFIG.HF_ENDPOINT

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
        self.driver: Optional[GraphDatabase.driver] = None

    def connect(self):
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
                self.driver.verify_connectivity()
                logger.info(f"✅ Connected to Neo4j at {self.uri}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Neo4j: {e}")
                self.driver = None

    def disconnect(self):
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        if not self.driver:
            raise ConnectionError("Not connected to Neo4j database")
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            records = [record.data() for record in result]
        return records

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HybridQueryEngine:
    
    def __init__(self):
        self.neo4j_conn = None
        self.query_engine = None
        self._setup_neo4j()
        self._setup_vector_search()

    def _setup_neo4j(self):
        try:
            self.neo4j_conn = Neo4jConnection()
            self.neo4j_conn.connect()
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            self.neo4j_conn = None
    
    def _setup_vector_search(self):
        try:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name = MY_CONFIG.EMBEDDING_MODEL
            )
            logger.info (f"✅ Using embedding model: {MY_CONFIG.EMBEDDING_MODEL}")

            # Initialize Milvus vector store
            vector_store = MilvusVectorStore(
                uri = MY_CONFIG.MILVUS_URI_HYBRID_GRAPH,  # Use Hybrid GraphRAG database
                dim = MY_CONFIG.EMBEDDING_LENGTH,
                collection_name = MY_CONFIG.COLLECTION_NAME, 
                overwrite=False
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            logger.info (f"✅ Connected to Hybrid GraphRAG Milvus instance: {MY_CONFIG.MILVUS_URI_HYBRID_GRAPH}")

            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, storage_context=storage_context)
            logger.info (f"✅ Loaded Hybrid GraphRAG index from: {MY_CONFIG.MILVUS_URI_HYBRID_GRAPH}")

            llm_model = MY_CONFIG.LLM_MODEL
            logger.info (f"✅ Using LLM model : {llm_model}")
            self.llm = LiteLLM(model=llm_model)
            Settings.llm = self.llm
            Settings.llm = self.llm
             
            self.query_engine = index.as_query_engine()
            
        except Exception as e:
            logger.error(f"❌ Vector setup error: {e}")
            self.query_engine = None

    def test_database_content(self) -> Dict:
        if not self.neo4j_conn or not self.neo4j_conn.driver:
            return {"error": "No Neo4j connection"}
            
        try:
            with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
                count_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = count_result.single()["node_count"]
                
                labels_result = session.run("MATCH (n) RETURN DISTINCT labels(n) as labels LIMIT 10")
                labels = [record["labels"] for record in labels_result]
                
                sample_result = session.run("""
                    MATCH (n) 
                    RETURN n.id as id, n.name as name, n.content as content, 
                           labels(n) as labels, keys(n) as properties
                    LIMIT 10
                """)
                samples = [dict(record) for record in sample_result]
                
                org_result = session.run("""
                    MATCH (n:Organization) 
                    RETURN n.id as id, n.name as name, n.content as content, labels(n) as labels
                    ORDER BY n.confidence DESC
                    LIMIT 10
                """)
                org_nodes = [dict(record) for record in org_result]
                
                rel_props_result = session.run("""
                    MATCH ()-[r]-()
                    RETURN DISTINCT keys(r) as rel_properties, type(r) as rel_type
                    LIMIT 10
                """)
                rel_properties = [dict(record) for record in rel_props_result]
                
                sample_rel_result = session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN a.name as from_name, type(r) as rel_type, b.name as to_name,
                           r.description as description, r.evidence as evidence, r.confidence as rel_confidence,
                           keys(r) as all_rel_props
                    LIMIT 10
                """)
                sample_relationships = [dict(record) for record in sample_rel_result]
                
                return {
                    "node_count": node_count,
                    "labels": labels,
                    "sample_nodes": samples,
                    "organization_nodes": org_nodes,
                    "relationship_properties": rel_properties,
                    "sample_relationships": sample_relationships
                }
                
        except Exception as e:
            return {"error": str(e)}

    def _get_graph_context_broader_search(self, query: str) -> str:
        """
        Broader search strategy that retrieves a sample of entities regardless of direct match
        Used as a last resort in the retry strategy
        
        Important: This method creates its own Neo4j session to avoid 'Session closed' errors
        """
        if not self.neo4j_conn or not self.neo4j_conn.driver:
            return ""
            
        try:
            logger.info("🔍 Trying broader search strategy: Random sampling of important entities")
            
            # Just get the most important entities in the database
            search_query = """
            MATCH (n) 
            WHERE n.confidence IS NOT NULL
            RETURN n.name as name, labels(n)[0] as type, n.content as content
            ORDER BY n.confidence DESC
            LIMIT 10
            """
            
            # Create a fresh session for this operation
            with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
                result = session.run(search_query)
                results = [dict(record) for record in result]
            
            if results:
                logger.info(f"✅ Broader search strategy successful: Found {len(results)} important entities")
                context_length = len(str(results))
                logger.info(f"📝 Generated graph context: {len(results)} entities, {context_length} characters")
                return str(results)[:8000]
            else:
                logger.error("❌ ALL RETRY STRATEGIES FAILED: Unable to retrieve graph context")
                logger.error("❌ This will cause the entire query to fail as graph context is required")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Broader search strategy error: {e}")
            return ""

    def _extract_entities_from_text(self, text: str, original_query: str) -> List[str]:
        """Extract key entities from vector search results to guide targeted graph search."""
        import re
        
        all_text = f"{original_query} {text}"
        entities = []
        
        # Extract capitalized words
        capitalized_patterns = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', all_text)
        entities.extend(capitalized_patterns)
        
        # Extract meaningful query words
        stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'the', 'and', 'are', 'is', 'of', 'to', 'in', 'for', 'with', 'that', 'this', 'does', 'do', 'any', 'run', 'have', 'has', 'a', 'an', 'at', 'by', 'be', 'been', 'was', 'were', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
        query_words = [w for w in original_query.split() if len(w) > 2 and w.lower() not in stop_words]
        entities.extend(query_words)
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', all_text)
        entities.extend(quoted_phrases)
        
        # Extract frequent words
        words = re.findall(r'\b\w{3,}\b', text.lower())
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        frequent_words = [word for word, freq in word_freq.items() if freq >= 2]
        entities.extend(frequent_words)
        
        # Extract organizational terms
        org_patterns = re.findall(r'\b\w*(?:group|team|company|corp|corporation|inc|org|organization|foundation|institute|university|college|school|center|centre|association|society|club|community|network|partnership|alliance|federation|union|agency|department|ministry|bureau|office|division|branch|unit)\w*\b', text, re.IGNORECASE)
        entities.extend(org_patterns)
        
        # Clean and deduplicate
        cleaned_entities = []
        for entity in entities:
            entity = entity.strip()
            if len(entity) > 1 and entity.lower() not in stop_words and entity not in cleaned_entities:
                cleaned_entities.append(entity)
        
        cleaned_entities.sort(key=len, reverse=True)
        return cleaned_entities[:12]

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Enhanced query analysis when vector extraction fails."""
        import re
        
        entities = []
        
        # Extract capitalized words
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(capitalized)
        
        # Extract meaningful query words
        stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'the', 'and', 'are', 'is', 'of', 'to', 'in', 'for', 'with', 'that', 'this', 'does', 'do', 'any', 'run', 'have', 'has', 'a', 'an', 'at', 'by', 'be', 'been', 'was', 'were', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
        meaningful_words = [w for w in query.split() if len(w) > 2 and w.lower() not in stop_words]
        entities.extend(meaningful_words)
        
        # Extract compound terms
        compound_terms = re.findall(r'\b\w+[-_]\w+\b', query)
        entities.extend(compound_terms)
        
        # Clean and deduplicate
        cleaned_entities = []
        for entity in entities:
            entity = entity.strip()
            if len(entity) > 1 and entity not in cleaned_entities:
                cleaned_entities.append(entity)
        
        return cleaned_entities[:10]

    def _extract_fact_triplets(self, entities_data: List[Dict], relationships_data: List[Dict]) -> List[Dict]:
        """Extract structured fact triplets (Subject-Predicate-Object) from graph data."""
        fact_triplets = []
        
        try:
            # Extract entity-based facts
            for entity in entities_data:
                name = entity.get("name", "")
                entity_type = entity.get("type", "")
                content = entity.get("content", "")
                confidence = entity.get("confidence", 0.5)
                
                if name and entity_type:
                    # Fact: Entity IS-A Type
                    fact_triplets.append({
                        "subject": name,
                        "predicate": "is_a",
                        "object": entity_type,
                        "confidence": confidence
                    })
                    
                    # Extract facts from content
                    if content:
                        # Pattern: "X is a Y that does Z"
                        import re
                        
                        # Generic relationship patterns
                        patterns = [
                            (r'is (?:a|an) ([^.]+)', "is_a"),
                            (r'works? (?:with|for|at) ([^.]+)', "associated_with"),
                            (r'develops? ([^.]+)', "develops"),
                            (r'supports? ([^.]+)', "supports"),
                            (r'manages? ([^.]+)', "manages"),
                            (r'creates? ([^.]+)', "creates"),
                            (r'provides? ([^.]+)', "provides"),
                            (r'focuses? on ([^.]+)', "focuses_on"),
                            (r'includes? ([^.]+)', "includes"),
                            (r'involves? ([^.]+)', "involves"),
                            (r'collaborates? with ([^.]+)', "collaborates_with"),
                            (r'partners? with ([^.]+)', "partners_with"),
                            (r'belongs? to ([^.]+)', "belongs_to"),
                            (r'part of ([^.]+)', "part_of")
                        ]
                        
                        for pattern, predicate in patterns:
                            matches = re.findall(pattern, content.lower())
                            for match in matches:
                                clean_object = match.strip().replace(',', '').replace('.', '')
                                if len(clean_object) > 2 and len(clean_object) < 100:
                                    fact_triplets.append({
                                        "subject": name,
                                        "predicate": predicate,
                                        "object": clean_object,
                                        "confidence": confidence * 0.8  # Lower confidence for extracted patterns
                                    })
            
            # Extract relationship-based facts
            for rel in relationships_data:
                rel_type = rel.get("rel_type", "")
                targets = rel.get("sample_targets", [])
                count = rel.get("count", 0)
                
                if rel_type and targets:
                    # Convert relationship type
                    predicate = rel_type.lower().replace("_", " ")
                    
                    for target in targets[:5]:  # Increased from 3 to 5 targets
                        if target:
                            fact_triplets.append({
                                "subject": "entities_in_context",
                                "predicate": predicate,
                                "object": target,
                                "confidence": min(0.9, count / 10.0)  # Scale confidence based on frequency
                            })
            
            # Remove duplicates
            unique_facts = []
            seen_facts = set()
            
            for fact in fact_triplets:
                fact_key = f"{fact['subject']}|{fact['predicate']}|{fact['object']}"
                if fact_key not in seen_facts:
                    seen_facts.add(fact_key)
                    unique_facts.append(fact)
            
            # Sort by confidence and limit
            unique_facts.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"📊 Extracted {len(unique_facts)} fact triplets from graph data")
            
            return unique_facts[:25]  # Increased from 15 to allow more facts
            
        except Exception as e:
            logger.error(f"❌ Error extracting fact triplets: {e}")
            return []

    def _summarize_facts(self, fact_triplets: List[Dict]) -> str:
        """Summarize extracted fact triplets into coherent statements."""
        if not fact_triplets:
            return ""
        
        try:
            # Group facts by subject for organization
            facts_by_subject = {}
            relationship_facts = []
            
            for fact in fact_triplets:
                subject = fact["subject"]
                predicate = fact["predicate"]
                obj = fact["object"]
                confidence = fact["confidence"]
                
                if subject == "entities_in_context":
                    relationship_facts.append(f"- {predicate.replace('_', ' ').title()}: {obj}")
                else:
                    if subject not in facts_by_subject:
                        facts_by_subject[subject] = []
                    
                    # Convert predicate to natural language
                    if predicate == "is_a":
                        fact_text = f"is a {obj}"
                    elif predicate.endswith("_with"):
                        fact_text = f"{predicate.replace('_', ' ')} {obj}"
                    else:
                        fact_text = f"{predicate.replace('_', ' ')} {obj}"
                    
                    facts_by_subject[subject].append((fact_text, confidence))
            
            # Build summary
            summary_parts = []
            
            # Add entity-specific facts
            for subject, facts in facts_by_subject.items():
                if len(facts) > 0:
                    # Sort by confidence
                    facts.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top facts for this subject
                    top_facts = [fact[0] for fact in facts[:5]]
                    
                    if len(top_facts) == 1:
                        summary_parts.append(f"• {subject} {top_facts[0]}")
                    else:
                        facts_text = ", ".join(top_facts[:-1]) + f", and {top_facts[-1]}"
                        summary_parts.append(f"• {subject} {facts_text}")
            
            # Add relationship facts
            if relationship_facts:
                summary_parts.append("• Key relationships:")
                summary_parts.extend(relationship_facts[:8])
            
            # Combine all parts
            if summary_parts:
                full_summary = "\n".join(summary_parts)
                logger.info(f"📝 Summarized {len(fact_triplets)} facts into {len(summary_parts)} statements")
                return full_summary
            else:
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error summarizing facts: {e}")
            return ""

    def _build_structured_prompt(self, query: str, vector_response: str, summarized_facts: str) -> str:
        """Build a structured prompt combining vector documents and summarized facts."""
        try:
            # Full hybrid prompt with both sources
            prompt = f"""You are answering a question using two complementary information sources. Please synthesize them into a comprehensive, coherent answer.

Question: {query}

SOURCE 1 - Document Information:
{vector_response}

SOURCE 2 - Knowledge Graph Facts:
{summarized_facts}

Instructions:
1. Synthesize information from both sources
2. Prioritize factual accuracy
3. Highlight relationships and connections when relevant
4. Provide a comprehensive answer that leverages both sources
5. If there are contradictions, note them
6. Be clear and well-structured

Please provide your synthesized answer:"""

            logger.info(f"📋 Built structured prompt: {len(prompt)} characters")
            return prompt
            
        except Exception as e:
            logger.error(f"❌ Error building structured prompt: {e}")
            # Fallback to simple combination
            return f"{vector_response}\n\nAdditional Context:\n{summarized_facts}"

    def _synthesize_final_answer(self, structured_prompt: str) -> str:
        """Use query engine approach for final synthesis."""
        try:
            if not hasattr(self, 'query_engine') or not self.query_engine:
                logger.warning("⚠️ Query engine not available, falling back to direct LLM")
                return self._fallback_llm_synthesis(structured_prompt)

            # Use the same query optimization
            optimized_prompt = query_utils.tweak_query(structured_prompt, MY_CONFIG.LLM_MODEL)
            
            logger.info("🔄 Synthesizing final answer with query engine...")
            synthesis_start = time.time()
            
            # Use query engine approach for consistency
            response = self.query_engine.query(optimized_prompt)
            
            synthesis_time = time.time() - synthesis_start
            logger.info(f"⏱️ Query engine synthesis completed in {synthesis_time:.2f} seconds")
            
            final_answer = str(response).strip()
            
            if final_answer and len(final_answer) > 10:
                logger.info(f"✅ Final answer synthesized: {len(final_answer)} characters")
                return final_answer
            else:
                logger.warning("⚠️ Query engine synthesis returned empty/short response, using fallback")
                return self._fallback_llm_synthesis(structured_prompt)
                
        except Exception as e:
            logger.error(f"❌ Error in query engine synthesis: {e}")
            logger.warning("⚠️ Falling back to direct LLM synthesis")
            return self._fallback_llm_synthesis(structured_prompt)

    def _fallback_llm_synthesis(self, structured_prompt: str) -> str:
        """Fallback method using direct LLM if query engine fails."""
        try:
            if not hasattr(self, 'llm') or not self.llm:
                logger.warning("⚠️ LLM not available for synthesis, returning structured prompt")
                return structured_prompt

            # Use query optimization
            optimized_prompt = query_utils.tweak_query(structured_prompt, MY_CONFIG.LLM_MODEL)
            
            # Use LLM to synthesize the final answer
            from llama_index.core.llms import ChatMessage
            
            messages = [
                ChatMessage(role="user", content=optimized_prompt)
            ]
            
            logger.info("🔄 Synthesizing final answer with direct LLM...")
            synthesis_start = time.time()
            
            response = self.llm.chat(messages)
            
            synthesis_time = time.time() - synthesis_start
            logger.info(f"⏱️ LLM synthesis completed in {synthesis_time:.2f} seconds")
            
            final_answer = str(response.message.content).strip()
            
            if final_answer and len(final_answer) > 10:
                logger.info(f"✅ Final answer synthesized: {len(final_answer)} characters")
                return final_answer
            else:
                logger.warning("⚠️ LLM synthesis returned empty/short response, using structured prompt")
                return structured_prompt
                
        except Exception as e:
            logger.error(f"❌ Error in LLM synthesis: {e}")
            logger.warning("⚠️ Falling back to structured prompt")
            return structured_prompt

    def _get_targeted_graph_context(self, query: str, extracted_entities: List[str]) -> Dict:
        """Get graph context using extracted entities for targeted search."""
        if not self.neo4j_conn or not self.neo4j_conn.driver:
            logger.error("❌ Neo4j driver not available for targeted graph context")
            return {"entities": [], "relationships": [], "raw_context": ""}
        
        # Use extracted entities or fall back to query words
        search_terms = extracted_entities if extracted_entities else [w for w in query.lower().split() if len(w) > 2][:8]
        
        if not search_terms:
            logger.warning("⚠️ No search terms available for targeted graph search")
            return {"entities": [], "relationships": [], "raw_context": ""}
        
        logger.info(f"🎯 Targeted graph search using entities: {search_terms[:8]}")  # Updated display limit
        
        # Maximum retry attempts
        max_retries = 3
        
        for retry_count in range(max_retries):
            try:
                # Enhanced search query for exact and partial matches
                search_query = """
                MATCH (n)
                WHERE ANY(term IN $search_terms WHERE 
                    toLower(n.name) CONTAINS toLower(term) OR 
                    toLower(n.content) CONTAINS toLower(term) OR
                    toLower(term) CONTAINS toLower(n.name))
                RETURN n.name as name, labels(n)[0] as type, n.content as content,
                       n.confidence as confidence,
                       CASE 
                         WHEN ANY(term IN $search_terms WHERE toLower(n.name) = toLower(term)) THEN 1.0
                         WHEN ANY(term IN $search_terms WHERE toLower(n.name) CONTAINS toLower(term)) THEN 0.8
                         WHEN ANY(term IN $search_terms WHERE toLower(n.content) CONTAINS toLower(term)) THEN 0.6
                         ELSE 0.3
                       END as match_score
                ORDER BY match_score DESC, COALESCE(n.confidence, 0) DESC
                LIMIT 25
                """
                
                # Variables to store results
                results = []
                rel_counts = []
                entity_types = {}
                
                # Create a fresh session for this attempt
                with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
                    # Execute the targeted search query
                    result = session.run(search_query, {"search_terms": search_terms})
                    results = [dict(record) for record in result]
                    
                    # Count entity types
                    for record in results:
                        entity_type = record.get("type", "Unknown")
                        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                    
                    # Get relationships for targeted entities
                    if results:
                        entity_names = [r.get("name") for r in results if r.get("name")]
                        if entity_names:
                            rel_query = """
                            MATCH (n)-[r]-(m)
                            WHERE n.name IN $entity_names
                            RETURN type(r) as rel_type, count(r) as count,
                                   collect(DISTINCT m.name)[0..5] as sample_targets
                            ORDER BY count(r) DESC
                            LIMIT 15
                            """
                            rel_result = session.run(rel_query, {"entity_names": entity_names})
                            rel_counts = [dict(record) for record in rel_result]
                
                # Log results
                entity_summary = []
                for entity_type, count in entity_types.items():
                    entity_summary.append(f"{count} {entity_type}(s)")
                
                if entity_summary:
                    logger.info(f"🎯 Targeted entities found: {', '.join(entity_summary)}")
                    
                    # Log sample entity names
                    sample_names = [r.get("name") for r in results[:5] if r.get("name")]
                    if sample_names:
                        logger.info(f"📄 Top targeted entities: {', '.join(sample_names)}")
                else:
                    logger.warning("⚠️ No targeted entities found in graph database")
                    # Try fallback with shorter terms
                    if retry_count < max_retries - 1:
                        shorter_terms = [term for term in search_terms if len(term) > 1][:8]
                        if shorter_terms:
                            logger.info(f"🔄 Retry with broader terms: {shorter_terms}")
                            search_terms = shorter_terms
                            continue
                    
                    # Final fallback to broader search
                    broader_result = self._get_graph_context_broader_search(query)
                    return {
                        "entities": [],
                        "relationships": [],
                        "raw_context": broader_result
                    }
                
                # Log relationship information
                if rel_counts:
                    rel_summary = []
                    for rel in rel_counts[:3]:
                        rel_type = rel.get('rel_type', 'unknown')
                        count = rel.get('count', 0)
                        targets = rel.get('sample_targets', [])
                        rel_summary.append(f"{count} {rel_type} → {', '.join(targets[:2])}")
                    
                    logger.info(f"🔗 Key relationships: {'; '.join(rel_summary)}")
                
                context_length = len(str(results))
                logger.info(f"📝 Generated targeted graph context: {len(results)} entities, {context_length} characters")
                
                # Return structured data
                return {
                    "entities": results,
                    "relationships": rel_counts,
                    "raw_context": str(results)[:8000]
                }
                    
            except Exception as e:
                logger.error(f"❌ Targeted graph search error on attempt {retry_count+1}: {e}")
                
                if retry_count < max_retries - 1:
                    logger.warning(f"🔄 Retry {retry_count+2}/{max_retries}: Targeted search failed, retrying...")
                    time.sleep(1)
        
        # If all retries failed
        logger.error(f"❌ All {max_retries} targeted graph search attempts failed")
        return {
            "entities": [],
            "relationships": [],
            "raw_context": ""
        }

    def close(self):
        if self.neo4j_conn and self.neo4j_conn.driver:
            self.neo4j_conn.disconnect()

    def run_query(self, query: str):
        logger.info("-----------------------------------")
        logger.info("🔍 HYBRID RAG QUERY PROCESS STARTED 🔍")
        start_time = time.time()
        
        # Import and apply query optimization
        query = query_utils.tweak_query(query, MY_CONFIG.LLM_MODEL)
        
        logger.info(f"\nProcessing Query:\n{query}")
        
        if not self.neo4j_conn or not self.neo4j_conn.driver:
            logger.error("❌ Hybrid RAG failed: Neo4j not connected.")
            return None
        
        try:
            self.neo4j_conn.driver.verify_connectivity()
            logger.info("✅ [1/7] Neo4j connection verified")
        except Exception as e:
            logger.error(f"❌ Hybrid RAG failed: Neo4j connection error: {e}")
            return None
        
        # STEP 1: VECTOR SEARCH (Primary)
        logger.info("🔄 [2/7] Starting Vector RAG retrieval (Primary)...")
        vector_start = time.time()
        vector_res = None
        extracted_entities = []
        
        if self.query_engine:
            try:
                vector_res = self.query_engine.query(query)
                vector_time = time.time() - vector_start
                
                # Log metadata about vector search
                if hasattr(vector_res, 'metadata') and vector_res.metadata.get("source_nodes"):
                    sources_count = len(vector_res.metadata.get("source_nodes"))
                    logger.info(f"📊 Vector Statistics: {sources_count} document chunks retrieved")
                else:
                    logger.info("📊 Vector Statistics: Response received (metadata format varies)")
                
                logger.info(f"⏱️ Vector RAG retrieval completed in {vector_time:.2f} seconds")
            except Exception as e:
                logger.error(f"❌ Hybrid RAG failed: Vector search error: {e}")
                return None
        else:
            logger.error("❌ Hybrid RAG failed: Vector engine not available.")
            return None
        
        # STEP 2: EXTRACT ENTITIES
        logger.info("🔄 [3/7] Extracting entities from vector results...")
        vector_text = str(vector_res)
        extracted_entities = self._extract_entities_from_text(vector_text, query)
        logger.info(f"🔍 Vector response length: {len(vector_text)} characters")
        
        if extracted_entities:
            logger.info(f"📄 Extracted {len(extracted_entities)} key entities: {extracted_entities[:5]}")
        else:
            logger.warning("⚠️ No entities extracted from vector results - using query analysis")
            extracted_entities = self._extract_entities_from_query(query)
            logger.info(f"📝 Query-based entities: {extracted_entities[:5]}")
        
        # STEP 3: TARGETED GRAPH SEARCH WITH RETRY LOGIC
        logger.info("🔄 [4/7] Starting targeted Graph RAG retrieval...")
        
        if not extracted_entities:
            logger.warning("⚠️ No entities available - using query analysis as fallback")
            extracted_entities = self._extract_entities_from_query(query)
        
        graph_start = time.time()
        
        # Retry graph search up to 3 times
        max_retries = 3
        entities_data = []
        relationships_data = []
        raw_context = ""
        
        for retry_attempt in range(max_retries):
            logger.info(f"🔍 Graph search attempt {retry_attempt + 1}/{max_retries}")
            graph_data = self._get_targeted_graph_context(query, extracted_entities)
            
            # Validate graph data
            entities_data = graph_data.get("entities", [])
            relationships_data = graph_data.get("relationships", [])
            raw_context = graph_data.get("raw_context", "")
            
            if entities_data or raw_context:
                logger.info(f"✅ Graph retrieval successful on attempt {retry_attempt + 1}")
                break
            else:
                logger.warning(f"⚠️ Attempt {retry_attempt + 1}/{max_retries} returned no graph data")
                if retry_attempt < max_retries - 1:
                    # Try with broader entity extraction on retry
                    if retry_attempt == 1:
                        extracted_entities = self._extract_entities_from_query(query)
                    time.sleep(0.5)  # Brief pause before retry
        
        graph_time = time.time() - graph_start
        logger.info(f"⏱️ Targeted Graph RAG retrieval completed in {graph_time:.2f} seconds")
        
        # If all retries failed, return error (NO VECTOR-ONLY FALLBACK)
        if not entities_data and not raw_context:
            logger.error("❌ All graph search attempts failed - GRAPH REQUIRED")
            logger.error("❌ This is a graph-dependent system - cannot proceed without graph data")
            return None
        
        logger.info(f"✅ Graph retrieval successful - found {len(entities_data)} entities")
        
        # STEP 4: EXTRACT FACT TRIPLETS
        logger.info("🔄 [5/7] Extracting fact triplets from graph data...")
        fact_start = time.perf_counter()
        fact_triplets = self._extract_fact_triplets(entities_data, relationships_data)
        fact_time = time.perf_counter() - fact_start
        
        logger.info(f"⏱️ Fact extraction completed in {fact_time:.4f} seconds")
        
        # STEP 5: SUMMARIZE FACTS
        logger.info("🔄 [6/7] Summarizing facts into coherent statements...")
        summary_start = time.perf_counter()
        summarized_facts = self._summarize_facts(fact_triplets)
        summary_time = time.perf_counter() - summary_start
        
        logger.info(f"⏱️ Facts summarization completed in {summary_time:.4f} seconds")
        
        # STEP 6: BUILD STRUCTURED PROMPT
        logger.info("🔄 [7/7] Building structured prompt and synthesizing final answer...")
        synthesis_start = time.time()
        
        structured_prompt = self._build_structured_prompt(query, str(vector_res), summarized_facts)
        
        # STEP 7: FINAL LLM SYNTHESIS
        final_answer = self._synthesize_final_answer(structured_prompt)
        
        synthesis_time = time.time() - synthesis_start
        logger.info(f"⏱️ Synthesis completed in {synthesis_time:.2f} seconds")
        
        # Final processing
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("-------"
                   + f"\nHybrid RAG Response (Complete Workflow):\n{final_answer}" 
                   + "\n\n⏱️ Complete Process Summary:"
                   + f"\n   - Vector RAG: {vector_time:.2f}s"
                   + f"\n   - Graph RAG: {graph_time:.2f}s"
                   + f"\n   - Fact Extraction: {fact_time:.4f}s"
                   + f"\n   - Facts Summarization: {summary_time:.4f}s"
                   + f"\n   - Final Synthesis: {synthesis_time:.2f}s"
                   + f"\n   - Total time: {total_time:.2f}s"
                   + f"\n   - Entities extracted: {len(extracted_entities)}"
                   + f"\n   - Facts extracted: {len(fact_triplets)}"
                   + (f"\n\nResponse Metadata:\n{json.dumps(vector_res.metadata, indent=2)}" if vector_res and hasattr(vector_res, 'metadata') else "")
                   )
        logger.info("✅ HYBRID RAG QUERY PROCESS COMPLETED ✅")
        logger.info("-----------------------------------")
        return final_answer

engine = None

def run_query(query: str):
    global engine
    if engine is None:
        logger.error("Engine not initialized")
        return None
    
    # Import and apply query optimization
    query = query_utils.tweak_query(query, MY_CONFIG.LLM_MODEL)
    
    result = engine.run_query(query)
    return result

engine = HybridQueryEngine()

if engine.neo4j_conn and engine.neo4j_conn.driver:
    print("🔍 Testing database content...")
    test_result = engine.test_database_content()
    print(f"Database test results: {test_result}")
    print("=" * 50)

logger.info("-----------------------------------")

if __name__ == "__main__":
    while True:
        user_query = input("\nEnter your question (or 'q' to exit): ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            logger.info("Goodbye!")
            break
        
        if user_query.strip() == "":
            continue
        
        try:
            run_query(user_query)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error processing query: {e}")