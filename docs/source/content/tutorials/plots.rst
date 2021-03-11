Creating Custom Plots
========================================

Overview
----------
To assign accurate labels to sources, it is important that user is presented with as full of a picture about each source as possible, therefore the ability to create specialised plots is paramount.

Simple
**********************************
astronomicAL allows for fast plot creation with the use of the :code:`create_plot` function which abstracts away the potentially complicated plotting code that is normally required and replaces it with a simple one-line call.

Optimised
**********************************
By default all custom plots have Datashader_ implemented, allowing millions of datapoints to be plotted at once whilst still remaining responsive.

.. _Datashader: http://holoviews.org/user_guide/Large_Data.html

Collaborative
**********************************
Allowing researchers to share their designs is pivotal for collaborative research. For this reason, we make it seamless to share your designs with others by automatically creating a settings page to allow new users to select which columns in their own datasets correspond with the naming convention of the author.

Custom Plots
--------------------------
Custom plots can be added as a new function to :doc:`astronomicAL.extensions.extension_plots <../apireference/extensions>`.

There are some requirements when declaring a new feature generation function:

1. The new function must have 2 input parameters:
  - :code:`data` - The dataframe containing the entire dataset.
  - :code:`selected` - The currently selected points (Default:None)
  - :code:`**kwargs` - keyword arguments for customising plots

2. The function must return the following:
  - :code:`plot` - The final plot to be rendered

3. The created function must be added in the form :code:`CustomPlot(new_plot_function, list_of_columns_used)` as a new value to the :code:`plot_dict` dictionary within the :code:`get_plot_dict` function, with a brief string key identifying the plot.

.. note::
    When using :code:`create_plot` within your custom plot functions, whenever you make use of a column name from your dataset you must reference it as a key from the :code:`config.settings` dictionary.

    For example: :code:`"x_axis_feature"` should always be written as :code:`config.settings["x_axis_feature"]`.

    If this is not done then you will likely cause a :code:`KeyError` to any external researchers who want to use your plot but have a different dataset with different column names.


Example: Plotting :math:`Y=X^2`
-----------------------------------
In this example we will show the simple case of creating a custom plot which shows :math:`Y=X^2` plotted along with the data.

.. code-block:: python
  :linenos:

  def x_squared(data, selected=None, **kwargs): # The function must include the parameters df, selected=None, and **Kwargs

      plot_data = create_plot(
          data,
          config.settings["x_axis"], # always use dataset column names as keys to config.settings
          config.settings["y_axis"],
      )

      line_x = np.linspace(
          np.min(data[config.settings["x_axis"]]),
          np.max(data[config.settings["x_axis"]]),
          30,
      ) # define the range of values for x

      line_y = np.square(line_x) # y=x^2

      line_data = pd.DataFrame(
          np.array([line_x, line_y]).T,
          columns=["x", "y"]
      ) # create dataframe with the newly created data

      line = create_plot(
          line_data,
          "x","y", # config.settings is not required here as these column names are not in reference to the main dataset
          plot_type="line", # we want a line drawn
          label_plot=False, # we don't need a legend for this data
          colours=False # Use default colours
      )

      x_squared_plot = plot_data * line # The * symbol combines multiple plots onto the same figure

      return x_squared_plot # The function must return the plot that is going to be rendered

Finally adding the new entry in the :code:`plot_dict` dictionary, **without specifying the parameters of the plotting function**:

.. code-block:: python

  def get_plot_dict():

      plot_dict = {
          "AGN Wedge": CustomPlot(
              agn_wedge, ["Log10(W3_Flux/W2_Flux)", "Log10(W2_Flux/W1_Flux)"]
          ),
          "BPT Plots": CustomPlot(
              bpt_plot,
              [
                  "Log10(NII_6584_FLUX/H_ALPHA_FLUX)",
                  "Log10(SII_6717_FLUX/H_ALPHA_FLUX)",
                  "Log10(OI_6300_FLUX/H_ALPHA_FLUX)",
                  "Log10(OIII_5007_FLUX/H_BETA_FLUX)",
              ],
          ),
          "X^2": CustomPlot(
              x_squared,
              ["x_axis", "y_axis"],
          ),
      }

      return plot_dict

And that is all that is required. The new :code:`x_squared` plot is now available to use in astronomicAL:

.. image:: ../../images/x_squared_in_plot_list.png

A settings page has automatically been generated, allowing users to selected which of their dataset columns correspond to the authors specified column.

.. image:: ../../images/x_squared_settings.png

Once the columns have been chosen, the user is presented with the brand new :code:`x_squared` plot:

.. image:: ../../images/x_squared_example.png

Optional plot flags
-------------------
