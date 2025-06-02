#### Dictionary with all data sharee between the extensions plots


shared_data = {"is_global": True}


# List of keys defined for the global shared_data dictionary
# is_global : bool,  debug key to see that the dictionary is seen from the dashboards
# Sparcl_client : object, The client to access NOIR-DataLab spectra
# DESI_coordinates : dictionary {"ra" : [ra], "dec" : [dec]}, coordinates of DESI spectra retrieved
# SDSS_coordinates : dictionary {"ra" : [ra], "dec" : [dec]}, coordinates of SDSS spectra retrieved
# Euclid_radius : float, radius of the Euclid cutout

subscribers = {}

panel_subscriptions = {}

def subscribe(key, callback):
    if key not in subscribers:
        subscribers[key] = []
    subscribers[key].append(callback)

def subscribe_with_panel_id(panel_id, key, callback):
    """Subscribe with panel tracking for cleanup"""
    subscribe(key, callback)
    if panel_id not in panel_subscriptions:
        panel_subscriptions[panel_id] = []
    panel_subscriptions[panel_id].append((key, callback))

def unsubscribe(key, callback):
    """Remove a callback from the subscribers for a given key"""
    if key in subscribers and callback in subscribers[key]:
        subscribers[key].remove(callback)
        if not subscribers[key]:
            del subscribers[key]

def cleanup_panel_subscriptions(panel_id):
    """Clean up all subscriptions for a specific panel"""
    if panel_id in panel_subscriptions:
        for key, callback in panel_subscriptions[panel_id]:
            unsubscribe(key, callback)
        del panel_subscriptions[panel_id]

def publish(key, value):
    shared_data[key] = value
    if key in subscribers:
        for callback in subscribers[key]:
            try:
                callback(value)
            except Exception as e:
                print(f"Error in callback for {key}: {e}")


  

#Trigger function = c = self.src.data.copy() ----> self.src.data = c