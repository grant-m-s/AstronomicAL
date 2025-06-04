#### Dictionary with all data sharee between the extensions plots and publish/subscribe architecture
import astronomicAL.config as config

shared_data = {"is_global": True}

radius = config.settings.get('Euclid_radius', 5.0)  
shared_data["Euclid_radius"] = radius


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
    #Subscribe with panel tracking
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
    """Remove all subscriptions for a specific panel ID"""
    if panel_id in panel_subscriptions:
        for key, callback in panel_subscriptions[panel_id]:
            unsubscribe(key, callback)
        del panel_subscriptions[panel_id]

def publish(key, value):
    shared_data[key] = value
    if key in subscribers:
        for i, callback in enumerate(subscribers[key]):
            panel_id = get_panel_id(callback)
            try:
                print(f"Calling N {i+1} callback function for {panel_id} due to a change in {key}")
                callback(value)
            except Exception as e:
                print(f"Error in callback for {key}: {e}")


def get_panel_id(target_callback):
    """for debug resons, given a callback returns the panel_id associated
      with that callback."""
    for panel_id, subscriptions in panel_subscriptions.items():
        for key, callback in subscriptions:
            if callback == target_callback:
                return panel_id
    return "Could not find the panel id"
    


#Trigger function = c = self.src.data.copy() ----> self.src.data = c