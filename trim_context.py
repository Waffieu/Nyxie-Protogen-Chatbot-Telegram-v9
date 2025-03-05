# Utility function to add to the UserMemory class in bot.py

def trim_context(self, user_id, reduce_percent=0.25):
    """
    Trim the user's message history to reduce memory usage while preserving
    the most important conversation context.
    
    Args:
        user_id (str): User ID
        reduce_percent (float): Percentage of messages to remove (0.0-1.0)
    """
    user_id = str(user_id)
    if user_id not in self.users:
        self.load_user_memory(user_id)
        
    messages = self.users[user_id].get("messages", [])
    if not messages:
        return
        
    # Calculate how many messages to keep
    num_messages = len(messages)
    num_to_remove = int(num_messages * reduce_percent)
    
    if num_to_remove < 1:
        num_to_remove = 1  # Remove at least one message
    
    # Strategy: Keep most recent messages and some oldest (historical context)
    # Remove messages from the middle (less relevant for current context)
    keep_recent = max(3, num_messages - num_to_remove)
    keep_oldest = min(2, num_messages - keep_recent)
    
    # Construct new message list with oldest + newest
    new_messages = messages[:keep_oldest] + messages[-keep_recent:]
    
    # Update user memory
    self.users[user_id]["messages"] = new_messages
    self.users[user_id]["total_tokens"] = sum(msg.get("tokens", 0) for msg in new_messages)
    
    # Save updated memory
    self.save_user_memory(user_id)
    
    return len(messages) - len(new_messages)  # Return number of messages removed
