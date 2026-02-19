
class ErrorTracker:
    """Tracks errors during astronomical source queries."""
    def __init__(self):
        self.reset()

    def log_error(self, error, error_message):
        self.has_error = True
        self.error = error 
        self.error_message = error_message
    
    def reset(self):
        self.has_error = False
        self.error = None
        self.error_message = ""

