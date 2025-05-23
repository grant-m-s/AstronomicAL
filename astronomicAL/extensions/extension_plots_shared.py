#### Dictionary with all data sharee between the extensions plots


shared_data = {"is_global": True}


# List of keys defined for the global shared_data dictionary
# is_global : bool,  debug key to see that the dictionary is seen from the dashboards
# Sparcl_client : object, The client to access NOIR-DataLab spectra
# DESI_coordinates : dictionary {"ra" : [ra], "dec" : [dec]}, coordinates of DESI spectra retrieved
# SDSS_coordinates : dictionary {"ra" : [ra], "dec" : [dec]}, coordinates of SDSS spectra retrieved
# Euclid_radius : float, radius of the Euclid cutout

subscribers = {}

def subscribe(key, callback):
    if key not in subscribers:
        subscribers[key] = []
    subscribers[key].append(callback)
 

def publish(key, value):
    shared_data[key] = value
    for callback in subscribers.get(key, []):
        callback(value)

  

#Trigger function = c = self.src.data.copy() ----> self.src.data = c