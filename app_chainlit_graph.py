import chainlit as cl
import os
import time
import asyncio
import re
import logging
import json
from typing import List, Dict, Optional, Any

# Import llama-index and related libraries
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from my_config import MY_CONFIG
from neo4j import GraphDatabase
import query_utils

# Global variables for engine and initialization status
engine = None
initialization_complete = False

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    force=True)

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
                logging.info(f"✅ Connected to Neo4j at {self.uri}")
            except Exception as e:
                logging.error(f"❌ Failed to connect to Neo4j: {e}")
                self.driver = None

    def disconnect(self):
        if self.driver:
            self.driver.close()
            self.driver = None
            logging.info("Disconnected from Neo4j")

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        if not self.driver:
            raise ConnectionError("Not connected to Neo4j database")
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            records = [record.data() for record in result]
        return records

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
            logging.error(f"❌ Failed to connect to Neo4j: {e}")
            self.neo4j_conn = None
    
    def _setup_vector_search(self):
        try:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name = MY_CONFIG.EMBEDDING_MODEL
            )
            logging.info(f"✅ Using embedding model: {MY_CONFIG.EMBEDDING_MODEL}")

            # Initialize Milvus vector store for Hybrid GraphRAG
            vector_store = MilvusVectorStore(
                uri = MY_CONFIG.MILVUS_URI_HYBRID_GRAPH,  # Use Hybrid GraphRAG database
                dim = MY_CONFIG.EMBEDDING_LENGTH,
                collection_name = MY_CONFIG.COLLECTION_NAME, 
                overwrite=False
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            logging.info(f"✅ Connected to Hybrid GraphRAG Milvus instance: {MY_CONFIG.MILVUS_URI_HYBRID_GRAPH}")
            
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, storage_context=storage_context)
            logging.info(f"✅ Loaded Hybrid GraphRAG index from: {MY_CONFIG.MILVUS_URI_HYBRID_GRAPH}")
            
            llm_model = MY_CONFIG.LLM_MODEL
            logging.info(f"✅ Using LLM model : {llm_model}")
            self.llm = LiteLLM(model=llm_model)
            Settings.llm = self.llm  # Also set global Settings
             
            self.query_engine = index.as_query_engine()
            
        except Exception as e:
            logging.error(f"❌ Vector setup error: {e}")
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

    def _extract_entities_from_text(self, text: str, original_query: str) -> List[str]:
        """
        Extract entities from vector search results to guide targeted graph search.
        This is the bridge between Vector RAG and Graph RAG.
        """
        
        # Combine original query words with text content
        all_text = f"{original_query} {text}"
        
        # Extract potential entities (capitalized words, multi-word phrases)
        entities = []
        
        # Extract capitalized words and phrases
        # Look for patterns like "Company Name", "Product Name", etc.
        capitalized_patterns = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', all_text)
        entities.extend(capitalized_patterns)
        
        # Extract meaningful words from query
        stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'the', 'and', 'are', 'is', 'of', 'to', 'in', 'for', 'with', 'that', 'this', 'does', 'do', 'any', 'run', 'have', 'has', 'a', 'an', 'at', 'by', 'be', 'been', 'was', 'were', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
        query_words = [w for w in original_query.split() if len(w) > 2 and w.lower() not in stop_words]
        entities.extend(query_words)
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', all_text)
        entities.extend(quoted_phrases)
        
        # Extract frequently mentioned terms
        words = re.findall(r'\b\w{3,}\b', text.lower())
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add words that appear 2+ times 
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
        
        # Prioritize longer entities
        cleaned_entities.sort(key=len, reverse=True)
        
        # Limit to top entities to avoid overwhelming graph search
        return cleaned_entities[:12]  

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Enhanced query analysis when vector extraction fails.
        Focus on extracting entities from the query itself.
        """
        
        entities = []
        
        # Extract capitalized words (proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(capitalized)
        
        # Extract query words (filtered stop words)
        stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'the', 'and', 'are', 'is', 'of', 'to', 'in', 'for', 'with', 'that', 'this', 'does', 'do', 'any', 'run', 'have', 'has', 'a', 'an', 'at', 'by', 'be', 'been', 'was', 'were', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
        meaningful_words = [w for w in query.split() if len(w) > 2 and w.lower() not in stop_words]
        entities.extend(meaningful_words)
        
        # Extract potential compound terms (hyphenated or multi-word concepts)
        compound_terms = re.findall(r'\b\w+[-_]\w+\b', query)
        entities.extend(compound_terms)
        
        # Clean and deduplicate
        cleaned_entities = []
        for entity in entities:
            entity = entity.strip()
            if len(entity) > 1 and entity not in cleaned_entities:
                cleaned_entities.append(entity)
        
        return cleaned_entities[:10] 

    def _get_targeted_graph_context(self, query: str, search_terms: List[str]) -> Dict:
        """
        Vector-First Approach: Use entities extracted from vector search to do targeted graph search.
        """
        if not self.neo4j_conn or not self.neo4j_conn.driver:
            logging.warning("⚠️ Neo4j driver not available for targeted graph search")
            return {"entities": [], "relationships": [], "raw_context": ""}
        
        max_retries = 3
        
        for retry_count in range(max_retries):
            try:
                if not search_terms:
                    logging.warning("⚠️ No search terms provided for targeted graph search")
                    return {"entities": [], "relationships": [], "raw_context": ""}
                
                logging.info(f"🎯 Targeted graph search with: {search_terms[:5]}")
                
                # Use targeted search based on extracted entities
                search_query = """
                MATCH (n)
                WHERE ANY(term IN $search_terms WHERE 
                    toLower(n.name) CONTAINS toLower(term) OR 
                    toLower(n.content) CONTAINS toLower(term) OR
                    toLower(n.id) CONTAINS toLower(term))
                RETURN n.id as id, n.name as name, n.content as content, 
                       labels(n)[0] as type, n.confidence as confidence
                ORDER BY n.confidence DESC
                LIMIT 15
                """
                
                results = []
                rel_counts = []
                entity_types = {}
                
                with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
                    result = session.run(search_query, {"search_terms": search_terms})
                    results = [dict(record) for record in result]
                    
                    # Count entity types
                    for record in results:
                        entity_type = record.get("type", "Unknown")
                        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                    
                    # Get relationships for these entities within the SAME session
                    if results:
                        entity_names = [r.get("name") for r in results if r.get("name")]
                        if entity_names:
                            rel_query = """
                            MATCH (n)-[r]-(m)
                            WHERE n.name IN $entity_names
                            RETURN type(r) as rel_type, count(r) as count,
                                   collect(DISTINCT m.name)[0..5] as sample_targets
                            ORDER BY count DESC
                            LIMIT 10
                            """
                            rel_result = session.run(rel_query, {"entity_names": entity_names})
                            rel_counts = [dict(record) for record in rel_result]
                
                entity_summary = []
                for entity_type, count in entity_types.items():
                    entity_summary.append(f"{count} {entity_type}(s)")
                
                if entity_summary:
                    logging.info(f"🎯 Targeted entities found: {', '.join(entity_summary)}")
                    
                    # Log sample entity names
                    sample_names = [r.get("name") for r in results[:5] if r.get("name")]
                    if sample_names:
                        logging.info(f"📄 Top targeted entities: {', '.join(sample_names)}")
                else:
                    logging.warning("⚠️ No targeted entities found in graph database")
                    # Try fallback with shorter terms
                    if retry_count < max_retries - 1:
                        shorter_terms = [term for term in search_terms if len(term) > 1][:8]
                        if shorter_terms:
                            logging.info(f"🔄 Retry with broader terms: {shorter_terms}")
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
                    
                    logging.info(f"🔗 Key relationships: {'; '.join(rel_summary)}")
                
                context_length = len(str(results))
                logging.info(f"📝 Generated targeted graph context: {len(results)} entities, {context_length} characters")
                
                # Return structured data for fact extraction
                return {
                    "entities": results,
                    "relationships": rel_counts,
                    "raw_context": str(results)[:8000]  # Increased from 3000 to allow fuller context
                }
                    
            except Exception as e:
                logging.error(f"❌ Targeted graph search error on attempt {retry_count+1}: {e}")
                
                if retry_count < max_retries - 1:
                    logging.warning(f"🔄 Retry {retry_count+2}/{max_retries}: Targeted search failed, retrying...")
                    time.sleep(1)
        
        # If all retries failed
        logging.error(f"❌ All {max_retries} targeted graph search attempts failed")
        return {
            "entities": [],
            "relationships": [],
            "raw_context": ""
        }

    def _get_graph_context_broader_search(self, query: str) -> str:
        """
        Broader search strategy that retrieves a sample of entities regardless of direct match
        Used as a last resort in the retry strategy
        """
        if not self.neo4j_conn or not self.neo4j_conn.driver:
            return ""
            
        try:
            logging.info("🔍 Trying broader search strategy: Random sampling of important entities")
            
            # Get top entities from database
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
                logging.info(f"✅ Broader search strategy successful: Found {len(results)} important entities")
                context_length = len(str(results))
                logging.info(f"📝 Generated graph context: {len(results)} entities, {context_length} characters")
                return str(results)[:8000]  # Increased from 3000 to allow fuller responses
            else:
                logging.error("❌ ALL RETRY STRATEGIES FAILED: Unable to retrieve graph context")
                logging.error("❌ This will cause the entire query to fail as graph context is required")
                return ""
                
        except Exception as e:
            logging.error(f"❌ Broader search strategy error: {e}")
            return ""

    def _extract_fact_triplets(self, entities_data: List[Dict], relationships_data: List[Dict]) -> List[Dict]:
        """
        Extract structured fact triplets (Subject-Predicate-Object) from graph data.
        """
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
                        
                        # Relationship patterns
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
                    # Convert relationship type to readable predicate
                    predicate = rel_type.lower().replace("_", " ")
                    
                    for target in targets[:5]:  
                        if target:
                            fact_triplets.append({
                                "subject": "entities_in_context",
                                "predicate": predicate,
                                "object": target,
                                "confidence": min(0.9, count / 10.0)  # Scale confidence based on frequency
                            })
            
            # Remove duplicates and sort by confidence
            unique_facts = []
            seen_facts = set()
            
            for fact in fact_triplets:
                fact_key = f"{fact['subject']}|{fact['predicate']}|{fact['object']}"
                if fact_key not in seen_facts:
                    seen_facts.add(fact_key)
                    unique_facts.append(fact)
            
            # Sort by confidence and limit
            unique_facts.sort(key=lambda x: x['confidence'], reverse=True)
            
            logging.info(f"📊 Extracted {len(unique_facts)} fact triplets from graph data")
            
            return unique_facts[:25]  # Increased from 15 to allow more facts
            
        except Exception as e:
            logging.error(f"❌ Error extracting fact triplets: {e}")
            return []

    def _summarize_facts(self, fact_triplets: List[Dict]) -> str:
        """
        Summarize extracted fact triplets into coherent, digestible statements.
        """
        if not fact_triplets:
            return ""
        
        try:
            # Group facts by subject for better organization
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
            
            # Add entity facts
            for subject, facts in facts_by_subject.items():
                if len(facts) > 0:
                    # Sort by confidence
                    facts.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top facts for this subject
                    top_facts = [fact[0] for fact in facts[:5]]  # Increased from 3 to 5
                    
                    if len(top_facts) == 1:
                        summary_parts.append(f"• {subject} {top_facts[0]}")
                    else:
                        facts_text = ", ".join(top_facts[:-1]) + f", and {top_facts[-1]}"
                        summary_parts.append(f"• {subject} {facts_text}")
            
            # Add relationship facts
            if relationship_facts:
                summary_parts.append("• Key relationships:")
                summary_parts.extend(relationship_facts[:8])  # Increased from 5 to 8 relationships
            
            # Combine all parts
            if summary_parts:
                full_summary = "\n".join(summary_parts)
                logging.info(f"📝 Summarized {len(fact_triplets)} facts into {len(summary_parts)} statements")
                return full_summary
            else:
                return ""
                
        except Exception as e:
            logging.error(f"❌ Error summarizing facts: {e}")
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

            logging.info(f"📋 Built structured prompt: {len(prompt)} characters")
            return prompt
            
        except Exception as e:
            logging.error(f"❌ Error building structured prompt: {e}")
            # Fallback to simple combination
            return f"{vector_response}\n\nAdditional Context:\n{summarized_facts}"

    def _synthesize_final_answer(self, structured_prompt: str) -> str:
        """
        Use query engine approach (same as vector-only) for final synthesis.
        This ensures consistent behavior between vector-only and graph workflows.
        """
        try:
            if not hasattr(self, 'query_engine') or not self.query_engine:
                logging.warning("⚠️ Query engine not available, falling back to direct LLM")
                return self._fallback_llm_synthesis(structured_prompt)

            # Use the query optimization
            optimized_prompt = query_utils.tweak_query(structured_prompt, MY_CONFIG.LLM_MODEL)
            
            logging.info("🔄 Synthesizing final answer with query engine (same as vector-only)...")
            synthesis_start = time.time()
            
            # Use query engine approach
            response = self.query_engine.query(optimized_prompt)
            
            synthesis_time = time.time() - synthesis_start
            logging.info(f"⏱️ Query engine synthesis completed in {synthesis_time:.2f} seconds")
            
            final_answer = str(response).strip()
            
            if final_answer and len(final_answer) > 10:
                logging.info(f"✅ Final answer synthesized via query engine: {len(final_answer)} characters")
                return final_answer
            else:
                logging.warning("⚠️ Query engine synthesis returned empty/short response, using fallback")
                return self._fallback_llm_synthesis(structured_prompt)
                
        except Exception as e:
            logging.error(f"❌ Error in query engine synthesis: {e}")
            logging.warning("⚠️ Falling back to direct LLM synthesis")
            return self._fallback_llm_synthesis(structured_prompt)

    def _fallback_llm_synthesis(self, structured_prompt: str) -> str:
        """
        Fallback method using direct LLM chat if query engine fails.
        """
        try:
            if not hasattr(self, 'llm') or not self.llm:
                logging.warning("⚠️ LLM not available for synthesis, returning structured prompt")
                return structured_prompt

            # Use the same query optimization as vector-only
            optimized_prompt = query_utils.tweak_query(structured_prompt, MY_CONFIG.LLM_MODEL)
            
            # Use LLM to synthesize the final answer
            from llama_index.core.llms import ChatMessage
            
            messages = [
                ChatMessage(role="user", content=optimized_prompt)
            ]
            
            logging.info("🔄 Synthesizing final answer with direct LLM (fallback)...")
            synthesis_start = time.time()
            
            response = self.llm.chat(messages)
            
            synthesis_time = time.time() - synthesis_start
            logging.info(f"⏱️ Direct LLM synthesis completed in {synthesis_time:.2f} seconds")
            
            final_answer = str(response).strip()
            
            if final_answer and len(final_answer) > 10:
                logging.info(f"✅ Final answer synthesized via direct LLM: {len(final_answer)} characters")
                return final_answer
            else:
                logging.error("❌ Direct LLM synthesis also failed")
                return "Sorry, I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            logging.error(f"❌ Error in direct LLM synthesis: {e}")
            return "Sorry, I encountered an error while processing your request."

    def close(self):
        if self.neo4j_conn and self.neo4j_conn.driver:
            self.neo4j_conn.disconnect()

    def run_query(self, query: str):
        logging.info("-----------------------------------")
        logging.info("🔍 HYBRID RAG QUERY PROCESS STARTED 🔍")
        start_time = time.time()
        
        # Import and apply query optimization
        query = query_utils.tweak_query(query, MY_CONFIG.LLM_MODEL)
        
        logging.info(f"\nProcessing Query:\n{query}")
        
        if not self.neo4j_conn or not self.neo4j_conn.driver:
            logging.error("❌ Hybrid RAG failed: Neo4j not connected.")
            return None
        
        try:
            self.neo4j_conn.driver.verify_connectivity()
            logging.info("✅ [1/7] Neo4j connection verified")
        except Exception as e:
            logging.error(f"❌ Hybrid RAG failed: Neo4j connection error: {e}")
            return None
        
        # STEP 1: VECTOR SEARCH (Primary)
        logging.info("🔄 [2/7] Starting Vector RAG retrieval (Primary)...")
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
                    logging.info(f"📊 Vector Statistics: {sources_count} document chunks retrieved")
                else:
                    logging.info("📊 Vector Statistics: Response received (metadata format varies)")
                
                logging.info(f"⏱️ Vector RAG retrieval completed in {vector_time:.2f} seconds")
            except Exception as e:
                logging.error(f"❌ Hybrid RAG failed: Vector search error: {e}")
                return None
        else:
            logging.error("❌ Hybrid RAG failed: Vector engine not available.")
            return None
        
        # STEP 2: EXTRACT ENTITIES
        logging.info("🔄 [3/7] Extracting entities from vector results...")
        vector_text = str(vector_res)
        extracted_entities = self._extract_entities_from_text(vector_text, query)
        logging.info(f"🔍 Vector response length: {len(vector_text)} characters")
        
        if extracted_entities:
            logging.info(f"📄 Extracted {len(extracted_entities)} key entities: {extracted_entities[:5]}")
        else:
            logging.warning("⚠️ No entities extracted from vector results - using query analysis")
            extracted_entities = self._extract_entities_from_query(query)
            logging.info(f"📝 Query-based entities: {extracted_entities[:5]}")
        
        # STEP 3: TARGETED GRAPH SEARCH (WITH WORKFLOW RETRIES)
        logging.info("🔄 [4/7] Starting targeted Graph RAG retrieval...")
        
        if not extracted_entities:
            logging.warning("⚠️ No entities available - using query analysis as fallback")
            extracted_entities = self._extract_entities_from_query(query)
        
        graph_start = time.time()
        graph_data = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(0.5)  # Small delay between retries
                    logging.info(f"🔄 Workflow retry attempt {attempt + 1}/{max_retries}")
                    if not extracted_entities:
                        extracted_entities = self._extract_entities_from_query(query)
                
                graph_data = self._get_targeted_graph_context(query, extracted_entities)
                
                # Validate graph data
                entities_data = graph_data.get("entities", [])
                relationships_data = graph_data.get("relationships", [])
                raw_context = graph_data.get("raw_context", "")
                
                if entities_data or raw_context:
                    break  # Success - exit retry loop
                else:
                    logging.warning(f"Attempt {attempt + 1}: No graph data retrieved")
                    
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        
        graph_time = time.time() - graph_start
        logging.info(f"⏱️ Targeted Graph RAG retrieval completed in {graph_time:.2f} seconds")
        
        if not entities_data and not raw_context:
            logging.error("❌ Hybrid RAG failed: No graph data retrieved after 3 workflow retries.")
            logging.error("❌ GRAPH-REQUIRED SYSTEM: No response without graph data")
            return None
        
        logging.info(f"✅ Graph retrieval successful - found {len(entities_data)} entities")
        
        # STEP 4: EXTRACT FACT TRIPLETS
        logging.info("🔄 [5/7] Extracting fact triplets from graph data...")
        fact_start = time.perf_counter()  # Use perf_counter for microsecond precision
        fact_triplets = self._extract_fact_triplets(entities_data, relationships_data)
        fact_time = time.perf_counter() - fact_start
        
        logging.info(f"⏱️ Fact extraction completed in {fact_time:.4f} seconds")
        
        # STEP 5: SUMMARIZE FACTS
        logging.info("🔄 [6/7] Summarizing facts into coherent statements...")
        summary_start = time.perf_counter()  # Use perf_counter for microsecond precision
        summarized_facts = self._summarize_facts(fact_triplets)
        summary_time = time.perf_counter() - summary_start
        
        logging.info(f"⏱️ Facts summarization completed in {summary_time:.4f} seconds")
        
        # STEP 6: BUILD STRUCTURED PROMPT
        logging.info("🔄 [7/7] Building structured prompt and synthesizing final answer...")
        synthesis_start = time.time()
        
        structured_prompt = self._build_structured_prompt(query, str(vector_res), summarized_facts)
        
        # STEP 7: FINAL LLM SYNTHESIS
        final_answer = self._synthesize_final_answer(structured_prompt)
        
        synthesis_time = time.time() - synthesis_start
        logging.info(f"⏱️ Synthesis completed in {synthesis_time:.2f} seconds")
        
        # Final processing
        end_time = time.time()
        total_time = end_time - start_time
        
        logging.info("-------"
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
        logging.info("✅ HYBRID RAG QUERY PROCESS COMPLETED ✅")
        logging.info("-----------------------------------")
        return final_answer

def extract_thinking_section(response_text):
    """
    Extract thinking section from LLM response if present.
    """
    thinking_pattern = r'<think>(.*?)</think>'
    match = re.search(thinking_pattern, response_text, re.DOTALL)
    
    if match:
        thinking_content = match.group(1).strip()
        cleaned_response = re.sub(thinking_pattern, '', response_text, flags=re.DOTALL).strip()
        return thinking_content, cleaned_response
    else:
        return None, response_text

def initialize():
    """
    Initialize Hybrid Query Engine with both vector and graph capabilities.
    This function sets up the necessary components for the chat application.
    """
    global engine, initialization_complete
    
    if initialization_complete:
        return
    
    logging.info("Initializing LLM, vector database, and Neo4j graph database...")
    
    try:
        engine = HybridQueryEngine()
        
        if engine.neo4j_conn and engine.neo4j_conn.driver and engine.query_engine:
            # Test database connectivity
            test_result = engine.test_database_content()
            if test_result.get("error"):
                logging.error(f"Database test failed: {test_result['error']}")
                raise Exception(f"Database test failed: {test_result['error']}")
            else:
                # Print graph-specific and common status messages
                print(f"✅ Database test successful: {test_result['node_count']} nodes found in Neo4j")
                print(f"✅ Connected to Neo4j at {MY_CONFIG.NEO4J_URI}")
                
                # Common print statements
                print("✅ Using embedding model: ", MY_CONFIG.EMBEDDING_MODEL)
                print("✅ LLM run environment: ", MY_CONFIG.LLM_RUN_ENV)
                print("✅ Using LLM model : ", MY_CONFIG.LLM_MODEL)
                print("✅ Connected to Milvus instance: ", MY_CONFIG.DB_URI)
                print("✅ Loaded index from vector db:", MY_CONFIG.DB_URI)
                
                logging.info("Successfully initialized LLM, vector database, and Neo4j graph database")
                initialization_complete = True
        else:
            error_msg = ""
            if not engine.neo4j_conn or not engine.neo4j_conn.driver:
                error_msg += "Neo4j connection failed. "
            if not engine.query_engine:
                error_msg += "Vector engine initialization failed. "
            
            logging.error(f"Error initializing hybrid engine: {error_msg}")
            raise Exception(error_msg)
            
    except Exception as e:
        initialization_complete = False
        error_msg = f"Error initializing LLM, vector database, or Neo4j graph database: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)

async def get_llm_response_with_graph(message):
    """
    Process the user message and get a response from the LLM with graph enhancement.
    Uses the initialized hybrid engine for semantic search and graph-based enhancements,
    showing the RAG process using Chainlit Steps.
    """
    global engine, initialization_complete
    
    # Check if hybrid engine is initialized
    if engine is None or not initialization_complete:
        return "System did not initialize. Please try again later.", 0
    
    # Import and apply query optimization
    message = query_utils.tweak_query(message, MY_CONFIG.LLM_MODEL)
    
    start_time = time.time()
    response_text = ''
    
    try:
        # Show detailed process steps
        async with cl.Step(name="🔍 Vector-First Hybrid GraphRAG Analysis", type="llm") as main_step:
            main_step.input = f"Query: {message}"
            main_step.output = "🔄 Processing with 7-step workflow:\n\n📚 Step 1: Vector document search\n🏷️ Step 2: Entity extraction\n🕸️ Step 3: Knowledge graph search\n📊 Step 4: Fact triplets extraction\n📝 Step 5: Facts summarization\n🧠 Step 6: Structured prompt building\n⚡ Step 7: LLM synthesis"
            
            # Process query with the engine
            loop = asyncio.get_event_loop()
            print(f"🔄 Starting query processing for: {message[:50]}...")
            result = await loop.run_in_executor(None, engine.run_query, message)
            print(f"🔄 Query processing completed. Result type: {type(result)}")
            
            # Check if result is None
            if result is None:
                logging.error("❌ Analysis failed, cannot provide a response")
                print("❌ Analysis returned None")
                main_step.output = "❌ Analysis failed: No graph data could be integrated."
                return "Sorry, I couldn't find relevant information to answer your question.", 0
            
            print(f"✅ Analysis successful. Response length: {len(str(result))}")
            main_step.output = f"✅ Vector-First Hybrid GraphRAG Complete!\n\n📊 Analysis Results:\n• Document search: Retrieved relevant content\n• Entity extraction: Identified key entities\n• Graph search: Found connected relationships\n• Fact analysis: Extracted structured triplets\n• Facts synthesis: Organized into coherent statements\n• Final response: Generated {len(str(result))} character answer\n\n🎯 Combined document content + knowledge graph insights!"
        
        # Format response
        if result:
            response_text = str(result).strip()
        
    except Exception as e:
        logging.error(f"Error getting LLM response: {str(e)}")
        response_text = f"Sorry, I encountered an error while processing your request:\n{str(e)}"
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Step 7: Process Summary (only if analysis succeeded)
    if response_text and not response_text.startswith("Sorry"):
        try:
            async with cl.Step(name="Analysis Complete", type="tool") as summary_step:
                summary_step.input = "Analysis summary"
                summary_step.output = f"✅ Successfully analyzed your question and provided a comprehensive answer ({len(response_text):,} characters) in {elapsed_time:.1f} seconds."
        except Exception as e:
            logging.error(f"Error creating summary step: {str(e)}")
    
    return response_text, elapsed_time

@cl.set_starters
async def set_starters():
    """Set starter questions for the Chainlit interface"""
    starters = []
    for prompt in MY_CONFIG.STARTER_PROMPTS:
        starters.append(
            cl.Starter(
                label=prompt.strip(),
                message=prompt.strip(),
            )
        )
    return starters

@cl.on_chat_start
async def start():
    """Initialize the application when a new chat starts"""
    try:
        # Initialize components
        initialize()
        
        welcome_message = """## Welcome to AllyCat! 🚀

Hi there! I can help you find information and answer your questions.

What would you like to know?
        """
        await cl.Message(content=welcome_message).send()
    except Exception as e:
        init_error = str(e)
        error_msg = f"""System Initialization Error

The system failed to initialize with the following error:

```
{init_error}
```

Please check your configuration and environment variables."""
        await cl.Message(content=error_msg).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    user_message = message.content
    print(f"📨 Received message: {user_message[:100]}...")
    
    try:
        # Get response from LLM with RAG steps shown FIRST
        print("🔄 Calling get_llm_response_with_graph...")
        response_text, elapsed_time = await get_llm_response_with_graph(user_message)
        print(f"✅ Got response: {len(response_text)} characters in {elapsed_time:.2f}s")
        
        thinking_content, cleaned_response = extract_thinking_section(response_text)
        
        # Add timing stat to response
        full_response = cleaned_response + f"\n\n⏱️ *Total time: {elapsed_time:.1f} seconds*"
        
        # THEN create a new message for streaming
        msg = cl.Message(content="")
        await msg.send()
        
        # Stream the response in small chunks for better UX
        chunk_size = 8  # Slightly larger chunks for better performance with graph responses
        for i in range(0, len(full_response), chunk_size):
            await msg.stream_token(full_response[i:i+chunk_size])
            await asyncio.sleep(0.005)  # Slightly faster to account for longer responses
        
        # Update the final message
        msg.content = full_response
        await msg.update()
        print("✅ Message streaming completed")
        
    except Exception as e:
        print(f"❌ Error in main message handler: {e}")
        import traceback
        traceback.print_exc()
        error_msg = cl.Message(content=f"Sorry, I encountered an error: {str(e)}")
        await error_msg.send()

## -------
if __name__ == '__main__':
    logging.info("App starting up...")