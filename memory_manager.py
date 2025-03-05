import os
import json
import logging
import numpy as np
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import google.generativeai as genai
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib
from sentence_transformers import SentenceTransformer
import torch
import time

logger = logging.getLogger(__name__)

class SemanticMemory:
    """
    Sophisticated semantic memory management system for bot conversations.
    Handles semantic indexing, retrieval, and memory optimization.
    """
    
    def __init__(self, memory_dir="semantic_memories", embedding_cache_file="embedding_cache.pkl", 
                 model_name="all-MiniLM-L6-v2"):
        self.memory_dir = memory_dir
        self.embedding_cache_file = os.path.join(memory_dir, embedding_cache_file)
        self.embedding_cache = {}
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.conversation_index = {}
        self.topic_clusters = defaultdict(list)
        self.last_save_time = time.time()
        self.save_interval = 60  # Save every 60 seconds
        
        # Try to load a local sentence transformer for better embeddings
        try:
            self.sentence_transformer = SentenceTransformer(model_name)
            self.use_transformer = True
            logger.info(f"Using SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer: {e}. Falling back to simpler method.")
            self.use_transformer = False
        
        # Ensure memory directory exists
        Path(memory_dir).mkdir(parents=True, exist_ok=True)
        
        # Load embedding cache if exists
        self.load_embedding_cache()
    
    def load_embedding_cache(self):
        """Load cached embeddings from disk"""
        try:
            if os.path.exists(self.embedding_cache_file):
                with open(self.embedding_cache_file, 'rb') as f:
                    try:
                        self.embedding_cache = pickle.load(f)
                        logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
                    except (pickle.PickleError, EOFError) as e:
                        logger.error(f"Error unpickling embedding cache: {e}. Creating new cache.")
                        self.embedding_cache = {}
            else:
                self.save_embedding_cache()  # Create empty cache file
                logger.info("Created new embedding cache file")
        except Exception as e:
            logger.error(f"Error loading embedding cache: {e}")
            self.embedding_cache = {}
    
    def save_embedding_cache(self, force=False):
        """Save embeddings cache to disk"""
        current_time = time.time()
        
        # Only save if forced or if enough time has passed since last save
        if not force and current_time - self.last_save_time < self.save_interval:
            return
            
        try:
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(self.embedding_cache_file), exist_ok=True)
            
            # Create a temp file first to prevent corruption if the process is interrupted
            temp_cache_file = f"{self.embedding_cache_file}.tmp"
            with open(temp_cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Replace the old cache with the new one
            if os.path.exists(self.embedding_cache_file):
                os.replace(temp_cache_file, self.embedding_cache_file)
            else:
                os.rename(temp_cache_file, self.embedding_cache_file)
                
            self.last_save_time = current_time
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving embedding cache: {e}")
    
    async def get_embedding(self, text, model):
        """Get embedding for text with caching for efficiency"""
        if not text:
            return np.zeros(384)  # Default dimension for empty text
            
        # Clean and normalize text for consistent embedding
        clean_text = self.clean_text(text)
        
        # Use a hash of the clean text as cache key
        cache_key = hashlib.md5(clean_text.encode('utf-8')).hexdigest()
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Use sentence-transformers if available (much better quality)
            if self.use_transformer:
                # Use the CPU to avoid CUDA memory issues
                with torch.no_grad():
                    embedding = self.sentence_transformer.encode([clean_text])[0]
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
            else:
                # Fallback to Gemini for embedding generation (slower but works without local models)
                embedding_prompt = f"""
                Task: Analyze this text and convert it into a semantic vector representation:
                
                "{clean_text}"
                
                Return ONLY a comma-separated list of 384 numbers between -1 and 1 that represents the semantic meaning of this text.
                The output should be ONLY the numbers, with no explanations or additional text.
                """
                
                response = await model.generate_content_async(embedding_prompt)
                text_response = response.text.strip()
                
                # Extract numbers from the response
                number_pattern = r'-?\d+\.?\d*'
                numbers = re.findall(number_pattern, text_response)
                
                if len(numbers) >= 100:  # At least get a reasonable number of dimensions
                    values = [float(n) for n in numbers[:384]]  # Limit to 384 dimensions
                    while len(values) < 384:
                        values.append(0.0)  # Pad if needed
                    embedding = np.array(values[:384])
                    embedding = embedding / np.linalg.norm(embedding)  # Normalize
                else:
                    # If extraction failed, create a fallback embedding using hash
                    hash_obj = hashlib.md5(clean_text.encode())
                    hash_bytes = hash_obj.digest()
                    embedding = np.array([((b / 255.0) * 2 - 1) for b in hash_bytes])
                    # Expand to 384 dimensions
                    while len(embedding) < 384:
                        embedding = np.concatenate([embedding, embedding])
                    embedding = embedding[:384]
            
            # Cache the embedding
            self.embedding_cache[cache_key] = embedding
            
            # Save cache periodically or when it reaches a threshold
            if len(self.embedding_cache) % 25 == 0:
                self.save_embedding_cache()
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Create a stable but unique fallback embedding for this text
            hash_obj = hashlib.md5(clean_text.encode())
            hash_bytes = hash_obj.digest()
            embedding = np.array([((b / 255.0) * 2 - 1) for b in hash_bytes])
            # Expand to 384 dimensions
            while len(embedding) < 384:
                embedding = np.concatenate([embedding, embedding])
            return embedding[:384]
    
    def clean_text(self, text):
        """Clean and normalize text for consistent embedding"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,?!;:\-\'"]', ' ', text)
        
        return text
    
    def extract_topics(self, text):
        """Extract key topics from text using advanced NLP techniques"""
        topics = []
        
        # Advanced keyword extraction
        if not text:
            return topics
            
        # Remove common words and keep significant terms
        words = re.findall(r'\b[a-zA-Z0-9]{4,}\b', text.lower())
        common_words = {'what', 'when', 'where', 'who', 'why', 'how', 'this', 'that', 'there', 'their', 'they', 'them',
                       'have', 'been', 'were', 'would', 'could', 'should', 'will', 'just', 'like', 'from', 'with',
                       'some', 'about', 'very', 'your', 'know', 'want', 'think', 'because', 'going'}
        
        # First pass: extract potential keywords
        potential_topics = [w for w in words if w not in common_words]
        
        # Find bigrams (two-word phrases) as they often contain more meaningful concepts
        if len(words) > 1:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            potential_topics.extend([b for b in bigrams if not any(word in common_words for word in b.split())])
        
        # Count topic frequency and keep most common
        topic_counts = defaultdict(int)
        for topic in potential_topics:
            topic_counts[topic] += 1
        
        # Sort by frequency, then by length (prefer longer topics)
        sorted_topics = sorted(topic_counts.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        topics = [topic for topic, _ in sorted_topics[:10]]  # Top 10 topics
        
        return topics[:5]  # Return top 5 topics
    
    def calculate_temporal_relevance(self, timestamp_str):
        """Calculate relevance score based on recency with a human-like memory decay function"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            now = datetime.now()
            hours_ago = (now - timestamp).total_seconds() / 3600
            
            # Human-like memory decay function (Ebbinghaus forgetting curve inspired)
            # - Recent memories (< 24h) are highly relevant
            # - Memories from 1-7 days decay moderately
            # - Older memories retain some importance but decay more quickly
            
            if hours_ago < 24:
                # Very recent (within 24 hours)
                return max(0.5, 1.0 - (hours_ago / 48))  # Gradual decay
            elif hours_ago < 168:  # Within a week
                # Days old (1-7 days)
                days = hours_ago / 24
                return 0.5 * np.exp(-0.1 * (days - 1))  # Moderate decay
            else:
                # Older than a week
                days = hours_ago / 24
                return 0.3 * np.exp(-0.05 * (days - 7))  # Slower decay for older memories
                
        except Exception as e:
            logger.error(f"Error calculating temporal relevance: {e}")
            return 0.3  # Default mid-low relevance if calculation fails
    
    def calculate_topic_relevance(self, message_topics, query_topics):
        """Calculate relevance based on topic overlap with semantic weighting"""
        if not message_topics or not query_topics:
            return 0.0
            
        # Calculate weighted Jaccard similarity for topics
        topic_set1 = set(message_topics)
        topic_set2 = set(query_topics)
        
        # Exact matches are most valuable
        intersection = topic_set1.intersection(topic_set2)
        exact_match_score = len(intersection) / max(len(topic_set1), len(topic_set2))
        
        # Partial matches (substrings) get partial credit
        partial_matches = 0
        for t1 in topic_set1:
            for t2 in topic_set2:
                # Skip exact matches (already counted)
                if t1 == t2:
                    continue
                    
                # Check for substring match in either direction
                if (t1 in t2) or (t2 in t1):
                    partial_matches += 0.5  # Half credit for partial match
                # Check for semantic similarity (e.g. 'dog' and 'canine')
                elif self.simple_word_similarity(t1, t2) > 0.7:
                    partial_matches += 0.3  # Partial credit for similar words
        
        # Combine exact and partial matches
        combined_score = exact_match_score + (partial_matches / (len(topic_set1) + len(topic_set2)))
        
        return min(1.0, combined_score)  # Cap at 1.0
    
    def simple_word_similarity(self, word1, word2):
        """Simple word similarity based on character overlap"""
        if not word1 or not word2:
            return 0.0
            
        # Simple character-level Jaccard similarity
        set1 = set(word1.lower())
        set2 = set(word2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_message_relevance(self, message, query_topics, query_embedding=None, current_time=None):
        """Calculate overall relevance score for a message using multiple factors"""
        if not current_time:
            current_time = datetime.now()
            
        # Extract components
        content = message.get('content', '')
        timestamp_str = message.get('timestamp', current_time.isoformat())
        
        # Extract topics from message content
        message_topics = self.extract_topics(content)
        
        # Calculate individual relevance scores
        temporal_score = self.calculate_temporal_relevance(timestamp_str)
        topic_score = self.calculate_topic_relevance(message_topics, query_topics)
        
        # Content relevance based on embedding similarity if available
        embedding_score = 0.0
        if query_embedding is not None and 'embedding' in message:
            try:
                message_embedding = np.array(message['embedding'])
                similarity = np.dot(query_embedding, message_embedding)
                embedding_score = max(0, similarity)  # Ensure non-negative
            except (ValueError, TypeError):
                embedding_score = 0.0
        
        # Length-based relevance (favor medium-length informative messages)
        content_length = len(content)
        if content_length < 10:
            length_score = 0.3  # Very short messages are less informative
        elif content_length < 100:
            length_score = 0.7  # Medium-short messages
        elif content_length < 500:
            length_score = 1.0  # Ideal length range
        else:
            length_score = 0.8  # Very long messages might have information but are less focused
        
        # Emotional significance (basic detection of emotionally charged content)
        emotional_words = {'love', 'hate', 'happy', 'sad', 'angry', 'excited', 'afraid', 'important',
                          'critical', 'emergency', 'urgent', 'special', 'wonderful', 'terrible'}
        emotional_count = sum(1 for word in content.lower().split() if word in emotional_words)
        emotion_score = min(1.0, emotional_count / 5)  # Cap at 1.0 (5 or more emotional words)
        
        # Combine scores with weights optimized for human-like memory
        combined_score = (
            0.25 * temporal_score +      # Recency
            0.35 * topic_score +         # Topical relevance (most important)
            0.15 * embedding_score +     # Semantic similarity
            0.15 * length_score +        # Information content
            0.10 * emotion_score         # Emotional significance
        )
        
        return combined_score

    async def analyze_query_intent(self, query, model):
        """Analyze the intent and context needs of the current query with sophisticated NLP"""
        try:
            intent_prompt = f"""
            Expert Analysis Task: Deeply analyze this user query to understand its intent, context needs, and implicit references.
            
            User Query: "{query}"
            
            Provide a detailed semantic analysis in this JSON format:
            {{
                "intent_type": "[question|statement|greeting|request|opinion|clarification|followup]",
                "requires_context": true/false,
                "topic_keywords": ["keyword1", "keyword2", ...],
                "time_sensitivity": "[high|medium|low]",
                "expected_references": ["specific reference 1", "specific reference 2", ...],
                "complexity": "[high|medium|low]",
                "is_followup_question": true/false,
                "emotional_tone": "[neutral|positive|negative|urgent|curious]",
                "conversation_phase": "[opening|ongoing|closing]"
            }}
            
            Be extremely precise and consider subtleties of language, implied references, and conversation flow.
            """
            
            response = await model.generate_content_async(intent_prompt)
            response_text = response.text
            
            # Extract JSON from response
            import json
            import re
            
            # Find JSON pattern in the response
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                try:
                    intent_data = json.loads(json_match.group(0))
                    return intent_data
                except json.JSONDecodeError:
                    logger.error("Failed to parse intent analysis JSON")
                
            # Fallback with default values but with extracted topics
            return {
                "intent_type": "question",
                "requires_context": True,
                "topic_keywords": self.extract_topics(query),
                "time_sensitivity": "medium",
                "expected_references": [],
                "complexity": "medium",
                "is_followup_question": self.is_likely_followup(query),
                "emotional_tone": "neutral",
                "conversation_phase": "ongoing"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return {
                "intent_type": "other",
                "requires_context": True,
                "topic_keywords": self.extract_topics(query),
                "time_sensitivity": "medium",
                "expected_references": [],
                "complexity": "medium",
                "is_followup_question": self.is_likely_followup(query),
                "emotional_tone": "neutral",
                "conversation_phase": "ongoing"
            }
            
    def is_likely_followup(self, query):
        """Determine if a query is likely a followup to a previous conversation"""
        query = query.lower()
        
        # Common follow-up indicators
        followup_indicators = [
            r'\b(and|but|so)\b', r'\bthen\b', r'\balso\b',
            r'^(what|how|why|when|where|who)\b', r'^is it\b',
            r'^(can|could) you\b', r'\bthat\b', r'\bthose\b',
            r'\bit\b', r'\bthey\b', r'\bthem\b', r'\bthis\b'
        ]
        
        # Check for pronouns and references without clear antecedents
        has_pronouns = re.search(r'\b(it|they|them|their|these|those|this|that)\b', query) is not None
        
        # Check for very short queries (often followups)
        is_short = len(query.split()) <= 4
        
        # Check for explicit followup patterns
        for pattern in followup_indicators:
            if re.search(pattern, query):
                return True
                
        # Combine heuristics
        return has_pronouns and is_short

    async def find_relevant_memories(self, user_id, current_query, messages, model, max_messages=10):
        """Find the most relevant memories for the current query using sophisticated relevance scoring"""
        if not messages:
            return []
            
        # Analyze query intent
        intent_data = await self.analyze_query_intent(current_query, model)
        query_topics = intent_data.get("topic_keywords", self.extract_topics(current_query))
        requires_context = intent_data.get("requires_context", True)
        is_followup = intent_data.get("is_followup_question", self.is_likely_followup(current_query))
        
        # If query appears to be a simple greeting or doesn't require context
        if not requires_context and not is_followup:
            # Return only recent messages for minimal context
            recent_msgs = messages[-3:]
            return [{
                "message": msg,
                "relevance": 1.0,  # Full relevance for recent messages
                "reason": "Recent conversation context"
            } for msg in recent_msgs]
        
        # Generate embedding for the query to calculate semantic similarity
        query_embedding = await self.get_embedding(current_query, model)
        
        # For context-requiring queries, perform full analysis
        relevant_memories = []
        
        # Two-phase memory retrieval (more human-like):
        # 1. Recent context (working memory)
        # 2. Relevant memories from longer-term memory
        
        # Phase 1: Recent context (like human working memory)
        recency_window = -5 if is_followup else -3  # Look back more if it's a followup question
        recent_context = messages[recency_window:] if abs(recency_window) < len(messages) else messages
        
        for msg in recent_context:
            # Generate embedding if not already cached
            if 'embedding' not in msg and 'content' in msg:
                msg['embedding'] = await self.get_embedding(msg['content'], model)
                
            # Boost relevance of recent messages in context
            relevance_score = self.calculate_message_relevance(
                msg, query_topics, query_embedding
            )
            
            # Higher base relevance for very recent messages in a followup
            if is_followup:
                relevance_score = relevance_score * 1.25  # 25% boost
            
            if relevance_score > 0.2:  # Threshold for relevance
                reason = "Recent conversation context"
                    
                relevant_memories.append({
                    "message": msg,
                    "relevance": min(relevance_score, 1.0),  # Cap at 1.0
                    "reason": reason
                })
        
        # Phase 2: Search for relevant memories in the broader history
        for msg in messages[:recency_window] if abs(recency_window) < len(messages) else []:
            # Generate embedding if not already cached
            if 'embedding' not in msg and 'content' in msg:
                msg['embedding'] = await self.get_embedding(msg['content'], model)
                
            relevance_score = self.calculate_message_relevance(
                msg, query_topics, query_embedding
            )
            
            # Add to relevant memories if score is above threshold
            if relevance_score > 0.4:  # Higher threshold for older messages
                # Determine reason for inclusion
                if relevance_score > 0.7:
                    reason = "Highly relevant to current query"
                elif "role" in msg and msg["role"] == "user":
                    reason = "Related user context"
                else:
                    reason = "Provides helpful background information"
                    
                relevant_memories.append({
                    "message": msg,
                    "relevance": relevance_score,
                    "reason": reason
                })
        
        # Sort by relevance and limit results
        relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Ensure diversity of memories by clustering if we have plenty
        if len(relevant_memories) > max_messages * 1.5:
            return self.diversify_memories(relevant_memories, max_messages)
        
        return relevant_memories[:max_messages]
        
    def diversify_memories(self, memories, max_count):
        """Ensure diversity in returned memories by using clustering techniques"""
        if len(memories) <= max_count:
            return memories
            
        selected_memories = []
        
        # Always include top memories
        top_count = max(1, max_count // 4)
        selected_memories.extend(memories[:top_count])
        remaining = memories[top_count:]
        
        # Cluster remaining memories by content
        clusters = defaultdict(list)
        
        for mem in remaining:
            content = mem["message"].get("content", "")
            topics = self.extract_topics(content)
            
            # Use first topic as cluster key, or "unknown" if none
            cluster_key = topics[0] if topics else "unknown"
            clusters[cluster_key].append(mem)
        
        # Take the highest relevance memory from each cluster
        for cluster_key, cluster_items in clusters.items():
            if len(selected_memories) < max_count and cluster_items:
                # Sort cluster by relevance
                cluster_items.sort(key=lambda x: x["relevance"], reverse=True)
                selected_memories.append(cluster_items[0])
        
        # If we still have room, add more memories by relevance
        if len(selected_memories) < max_count:
            # Flatten remaining memories and sort by relevance
            remaining_flat = [m for cluster in clusters.values() for m in cluster[1:] if m not in selected_memories]
            remaining_flat.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Add until we reach max_count
            selected_memories.extend(remaining_flat[:max_count - len(selected_memories)])
        
        # Sort final selection by relevance
        selected_memories.sort(key=lambda x: x["relevance"], reverse=True)
        
        return selected_memories[:max_count]
        
    def identify_conversation_segments(self, messages):
        """Identify distinct conversation segments in message history"""
        if not messages:
            return []
            
        segments = []
        current_segment = []
        
        # Time threshold for new segment (adaptive)
        time_threshold = 30 * 60  # 30 minutes in seconds (default)
        
        for i, msg in enumerate(messages):
            # Start new segment if this is first message
            if i == 0:
                current_segment = [msg]
                continue
                
            # Get timestamps
            try:
                prev_time = datetime.fromisoformat(messages[i-1].get('timestamp', ''))
                curr_time = datetime.fromisoformat(msg.get('timestamp', ''))
                
                # Calculate time difference
                time_diff = (curr_time - prev_time).total_seconds()
                
                # Check topic shift
                prev_topics = self.extract_topics(messages[i-1].get('content', ''))
                curr_topics = self.extract_topics(msg.get('content', ''))
                topic_overlap = self.calculate_topic_relevance(prev_topics, curr_topics)
                topic_shift = topic_overlap < 0.3  # Low topic overlap indicates shift
                
                # Check if this is a new segment based on time gap or topic shift
                if time_diff > time_threshold or topic_shift:
                    # Save current segment and start new one
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = [msg]
                else:
                    # Continue current segment
                    current_segment.append(msg)
            except Exception as e:
                # Fallback on errors
                logger.error(f"Error in conversation segmentation: {e}")
                current_segment.append(msg)
        
        # Add final segment
        if current_segment:
            segments.append(current_segment)
            
        return segments
    
    def reconstruct_context(self, relevant_memories, current_query, max_tokens=4096):
        """Reconstruct conversation context from relevant memories in a human-readable format"""
        if not relevant_memories:
            return ""
            
        # Start with most relevant memories
        context_parts = []
        token_count = 0
        
        # Track added message IDs to avoid duplicates
        added_msgs = set()

        # Group memories by relevance tier
        high_relevance = []
        medium_relevance = []
        low_relevance = []
        
        for memory in relevant_memories:
            rel_score = memory["relevance"]
            if rel_score > 0.7:
                high_relevance.append(memory)
            elif rel_score > 0.4:
                medium_relevance.append(memory)
            else:
                low_relevance.append(memory)
        
        # Sort within each tier by timestamp to preserve conversation flow
        for tier in [high_relevance, medium_relevance, low_relevance]:
            tier.sort(key=lambda x: x["message"].get("timestamp", ""))
        
        # Process high relevance memories first (critical context)
        for memory in high_relevance:
            msg = memory["message"]
            msg_id = msg.get('timestamp', '') + msg.get('content', '')[:20]
            
            if msg_id in added_msgs:
                continue
                
            # Estimate tokens (rough estimation: 4 chars ~= 1 token)
            estimated_tokens = len(msg.get('content', '')) // 4
            
            if token_count + estimated_tokens <= max_tokens:
                role = "User" if msg.get('role') == 'user' else "Assistant"
                context_parts.append(f"{role}: {msg.get('content', '')}")
                token_count += estimated_tokens
                added_msgs.add(msg_id)
        
        # Add medium relevance memories next (supporting context)
        for memory in medium_relevance:
            msg = memory["message"]
            msg_id = msg.get('timestamp', '') + msg.get('content', '')[:20]
            
            if msg_id in added_msgs:
                continue
                
            # Estimate tokens
            estimated_tokens = len(msg.get('content', '')) // 4
            
            if token_count + estimated_tokens <= max_tokens:
                role = "User" if msg.get('role') == 'user' else "Assistant"
                context_parts.append(f"{role}: {msg.get('content', '')}")
                token_count += estimated_tokens
                added_msgs.add(msg_id)
        
        # Add low relevance memories last (if space available)
        for memory in low_relevance:
            msg = memory["message"]
            msg_id = msg.get('timestamp', '') + msg.get('content', '')[:20]
            
            if msg_id in added_msgs:
                continue
                
            # Estimate tokens
            estimated_tokens = len(msg.get('content', '')) // 4
            
            if token_count + estimated_tokens <= max_tokens:
                role = "User" if msg.get('role') == 'user' else "Assistant"
                context_parts.append(f"{role}: {msg.get('content', '')}")
                token_count += estimated_tokens
                added_msgs.add(msg_id)
        
        # Re-sort all context parts by timestamp for proper conversational flow
        # This requires parsing timestamps from messages first
        timestamp_dict = {}
        for i, part in enumerate(context_parts):
            msg_id = None
            for memory in relevant_memories:
                msg = memory["message"]
                if part.endswith(msg.get('content', '')):
                    msg_id = msg.get('timestamp', '')
                    break
            timestamp_dict[i] = msg_id or f"{i}"  # Fallback to index if no timestamp
        
        # Sort context parts by timestamp
        sorted_indices = sorted(range(len(context_parts)), key=lambda i: timestamp_dict[i])
        context_parts = [context_parts[i] for i in sorted_indices]
        
        # Format context for the AI
        formatted_context = "\n\n".join(context_parts)
        
        # Add explanation header for the AI
        header = f"""The following are semantically relevant messages from the conversation history, 
selected because they relate to the user's current message: "{current_query}"

"""
        return header + formatted_context

    def get_memory_stats(self):
        """Get statistics about the memory system"""
        stats = {
            "embedding_cache_size": len(self.embedding_cache),
            "cache_file_size_kb": self._get_file_size_kb(self.embedding_cache_file),
            "last_save_time": datetime.fromtimestamp(self.last_save_time).isoformat(),
            "transformer_model": "using sentence_transformer" if self.use_transformer else "using fallback method"
        }
        return stats
        
    def _get_file_size_kb(self, filepath):
        """Get file size in KB"""
        try:
            if os.path.exists(filepath):
                return os.path.getsize(filepath) / 1024
            return 0
        except Exception:
            return 0
