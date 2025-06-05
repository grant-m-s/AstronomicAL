import astronomicAL.config as config

# List of keys defined for the global shared_data dictionary
# is_global : bool,  debug key to see that the dictionary is seen from the dashboards
# Sparcl_client : object, The client to access NOIR-DataLab spectra
# DESI_coordinates : dictionary {"ra" : [ra], "dec" : [dec]}, coordinates of DESI spectra retrieved
# SDSS_coordinates : dictionary {"ra" : [ra], "dec" : [dec]}, coordinates of SDSS spectra retrieved
# Euclid_radius : float, radius of the Euclid cutout




class SharedDataManager:

    def __init__(self, **kwargs):
        
        self.data = {}
        self.data.update(kwargs)
        self.subscriptions = {} # key : (panel_id : callback)
        self.publishers = {}    # panel_id : key
        self.key_publishers = {} # key : panel_id 
        self.extension_panels = {}

    
    def publish(self, panel_id, key ,value):
        self.data[key] = value
        
        #keeps track of publishers so that i can clean their published values when panels are closed
        if panel_id not in self.publishers:
            self.publishers[panel_id] = set()
        self.publishers[panel_id].add(key)
        
        #keeps track of panels using the same keys, we clean the dictionary 
        #only if the key is not being published by another panel.
        #This could happen if there are 2+ extension plots showing the same stuff
        if key not in self.key_publishers:
            self.key_publishers[key] = set()
        self.key_publishers[key].add(panel_id)
        
        if key in self.subscriptions:
            for i, (subscriber_panel_id, callback) in enumerate(self.subscriptions[key]):
                if subscriber_panel_id != panel_id:
                    try:
                        callback(value)
                        print(f"Callback function N {i+1} due to a change in {key}")
                    except Exception as e:
                        print(f"Error notifying {subscriber_panel_id}: {e}")
    
    def subscribe(self, panel_id, key, callback):
        if key not in self.subscriptions:
            self.subscriptions[key] = []
        self.subscriptions[key].append((panel_id, callback))

    def unsubscribe(self, panel_id, key):
        if key in self.subscriptions:
            self.subscriptions[key] = [
            (sub_panel_id, callback) for sub_panel_id, callback in self.subscriptions[key]
            if sub_panel_id != panel_id
            ]
            if not self.subscriptions[key]:
                del self.subscriptions[key]

    def unsubscribe_panel(self, panel_id):
        for key in list(self.subscriptions):
            self.unsubscribe(panel_id, key)

    
    def clean_published_from_panel(self, panel_id):
        if panel_id in self.publishers:
            for key in self.publishers[panel_id]:
                if key in self.key_publishers:
                    self.key_publishers[key].discard(panel_id)
                    
                    # Only delete the key if no other panels are using it
                    if not self.key_publishers[key]:
                        if key in self.data:
                            del self.data[key]
                        del self.key_publishers[key]
            del self.publishers[panel_id]
            
    
    def cleanup_extension_panel(self, panel_id):
        #Unsubscribe and clean published values
        self.clean_published_from_panel(panel_id)
        self.unsubscribe_panel(panel_id)
    
    def get_panel_id_for_callback(self, target_callback):
        """For debug: returns the panel_id associated with the given callback."""
        for key, subscribers in self.subscriptions.items():
            for panel_id, callback in subscribers:
                if callback == target_callback:
                    return panel_id
        return "Could not find the panel id"
    
    def is_subscribed(self, panel_id, key):
        return any(pid == panel_id for pid, _ in self.subscriptions.get(key, []))

    
    def set_data(self, key, value):
        self.data[key] = value
    
    def get_data(self, key, default=None):
        return self.data.get(key, default)
    
    def replace_subscribe(self, panel_id, key, new_callback):
        if key not in self.subscriptions:
            self.subscribe(panel_id, key, new_callback)
            return

        # Replace the callback for the given panel_id
        updated_subscriptions = []
        replaced = False
        for sub_panel_id, callback in self.subscriptions[key]:
            if sub_panel_id == panel_id:
                updated_subscriptions.append((panel_id, new_callback))
                replaced = True
            else:
                updated_subscriptions.append((sub_panel_id, callback))

        self.subscriptions[key] = updated_subscriptions

        if not replaced:
            self.subscribe(panel_id, key, new_callback)
        else:
            print(f"Subscription for panel_id '{panel_id}' and key '{key}' replaced successfully.")
    
    
shared_data = SharedDataManager(is_global = True)
shared_data.set_data("Euclid_radius",  config.settings.get('Euclid_radius', 5.0))

