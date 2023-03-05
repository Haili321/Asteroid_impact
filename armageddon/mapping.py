import os
import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster


def plot_circle(lat, lon, radius, color=None, map=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).
    Parameters
    ----------
    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object
    Returns
    -------
    Folium map object
    Examples
    --------
    >>> import folium
    >>> res = plot_circle(52.79, -2.95, 1e3, map=None)
    >>> isinstance(res, folium.Map)
    True
    """

    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True)

    folium.Circle([lat, lon], radius, fill=True, color=color,
                  fillOpacity=0.6, **kwargs).add_to(map)

    return map


def plot_results(lat_entry, lon_entry, lat_blast, lon_blast, radii,
                 postcodes=None, pop_size=None, sector=False, **kwargs):
    """
    Plots a map, with the entry point, the blast point and the
    different levels. Additionaly postcodes, popuation size can
    be passed to the function. If this is done, it is also
    necessary to specify whether those are in the sectors
    format.
    Parameters
    ----------
    lat_entry: float
        latitude of entry point (degrees)
    lon_entry: float
        longitude of entry point (degrees)
    lat_blast: float
        latitude of blast point (degrees)
    lon_blast: float
        longitude of blast point (degrees)
    radii: float
        diffrent radii for every level
    postcodes: list of lists
        effected postcodes for every level
    pop_size: list of lists
        effected population for every postcode in every level
    sector: boolean
        whether it is the sector format
    Returns
    -------
    Folium map object
    Examples
    --------
    >>> import folium
    >>> lat_entry = 52.655394701839896
    >>> lon_entry = -1.3030671874328144
    >>> lat_blast = 53.64040971412841
    >>> lon_blast = -1.2523687602455351
    >>> radii = [37115.469444321236, 11994.891298340393,\
    4654.807146118714, 2206.0544987188114]
    >>> res = plot_results(lat_entry, lon_entry,\
    lat_blast, lon_blast, radii)
    >>> isinstance(res, folium.Map)
    True
    """

    map_plot = None

    colors_names = ['orange', 'red', 'darkred', 'black']

    # Plotting zones
    for idx, rad in enumerate(radii):
        map_plot = plot_circle(lat_blast, lon_blast, rad,
                               colors_names[idx], map_plot)

    # Plotting flight path
    folium.PolyLine([
        [lat_entry, lon_entry],
        [lat_blast, lon_blast]
    ], color='#000', dash_array='10').add_to(map_plot)

    # Entry Point
    folium.Marker([lat_entry, lon_entry],
                  color='#000',
                  tooltip='Entry Point:<br>Latitude: {}<br>'
                  'Lonitude: {}'.format(lat_entry,
                                        lon_entry)).add_to(map_plot)

    # Blast point
    folium.Marker([lat_blast, lon_blast],
                  color='#000',
                  tooltip='Blast Point:<br>Latitude: {}<br>'
                  'Lonitude: {}'.format(lat_blast,
                                        lon_blast)).add_to(map_plot)

    # read data
    data_path = os.sep.join((os.path.dirname(__file__), '..',
                             'resources',
                             'full_postcodes.csv'))

    geo_data = pd.read_csv(data_path, usecols=[
                           'Postcode', 'Latitude', 'Longitude'])

    # Dont plot markers twice
    already_seen = np.array([])

    if postcodes is not None and pop_size is not None:
        for row_idx in range(len(postcodes)-1, -1, -1):
            row = postcodes[row_idx]

            # Icon create function taken from
            # https://github.com/Extralait/UniMap/tree/c04073fd10897809b3236b3838009579cf927d34
            ic_f = '''
            function (cluster) {
                var childCount = cluster.getChildCount();
                return new L.DivIcon({ html: '<style type="text/css">'''
            ic_f += f'.my_new_icon_{row_idx}'
            ic_f += '{display:flex;justify-content:center;'
            ic_f += 'border: white solid 2px;background-color:{};'\
                .format(colors_names[row_idx])
            ic_f += '''border-radius:25px;align-items:center;}</style>'''
            ic_f += '''<div style="color: white">'''
            ic_f += '''<span>' + childCount + '</span></div>', className: '''
            ic_f += f'"my_new_icon_{row_idx}"'
            ic_f += ''', iconSize: new L.Point(30, 30) });
                }'''

            # Using Cluster for performance
            cluster = MarkerCluster(
                name='Level {}'.format(len(postcodes) - row_idx),
                icon_create_function=ic_f).add_to(map_plot)

            row_df = pd.DataFrame(row, columns=['Postcode'])
            row_df['Population'] = pop_size[row_idx]

            # sectors as input
            if sector:

                # convert to sector postcode and
                # calculate mean lat and lon from
                # every unit in one sector to
                # display marker.
                geo_data['Postcode'] = pd.Series(
                    geo_data['Postcode'].values.astype('U5'))
                grouped_df = geo_data[['Postcode', 'Latitude',
                                       'Longitude']].groupby('Postcode').mean()

                # get lat lon data for postcode
                merged_df = row_df.merge(grouped_df, on='Postcode')
                merged_df = merged_df.loc[~merged_df['Postcode'].isin(
                    already_seen)]

            # units as input
            else:

                # get lat lon data for postcode
                merged_df = row_df.merge(geo_data, on='Postcode').fillna(0)
                merged_df = merged_df.loc[~merged_df['Postcode'].isin(
                    already_seen)]

            # display marker and add it to thecluster
            for idx in range(len(merged_df)):
                folium.Marker(
                    [merged_df.iloc[idx]['Latitude'],
                     merged_df.iloc[idx]['Longitude']],
                    colors=colors_names[row_idx],
                    icon=folium.Icon(
                        color=colors_names[row_idx],
                        icon='map-marker',
                        prefix='fa'),
                    tooltip='Postcode: {}<br>Population: {}'
                    '<br>Latitude: {}<br>'
                    'Longitude: {}'.format(merged_df.iloc[idx]['Postcode'],
                                           merged_df.iloc[idx]['Population'],
                                           merged_df.iloc[idx]['Latitude'],
                                           merged_df.iloc[idx]['Longitude']))\
                    .add_to(cluster)

            # Adding all postcodes to already seen
            already_seen = np.append(
                already_seen, merged_df['Postcode'].values)

        # Adding Layer Controlls for Clusters
        folium.LayerControl().add_to(map_plot)

    return map_plot
