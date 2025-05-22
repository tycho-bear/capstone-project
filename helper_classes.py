import math
import numpy
import numpy as np
import copy
import matplotlib.pyplot as plot


# tour class



# city class
# name, position
# maybe indicate whether these are lat/long coordinates?

class City:
    """
    Class representing a city. It has a name and two x/y or lat/lon coordinates.
    """
    def __init__(self, name: str, x: float, y: float) -> None:
        """
        Creates a new City.

        :param name: (str) The name of this City, such as "Seattle".
        :param x: (float) The x coordinate (alternatively, latitude) of this
            City.
        :param y: (float) The y coordinate (alternatively, longitude) of this
            City.
        """
        self.name = name
        self.x = x
        self.y = y

    def distance_to(self, other_city: "City", use_geographical_distance=False):
        """
        Calculates the distance between this City and another City. Set
        ``use_geographical_distance`` = ``True`` to calculate the distance along
        Earth's surface.

        :param other_city: (City) The other City object to calculate the
            distance to.
        :return: (float) the distance to the other City.
        """
        if not use_geographical_distance:
            # use Euclidean distance instead
            sum_term = (self.x - other_city.x)**2 + (self.y - other_city.y)**2
            return math.sqrt(sum_term)

        # TODO: implement geographical distance
        # https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
        earth_radius = 6371.009  # kilometers

    def __str__(self):
        """
        This is used for printing.

        :return: (str) A string representation of this City object.
        """
        return f"City \"{self.name}\" at ({self.x}, {self.y})"


class Tour:
    """
    Class representing a tour of cities in the traveling salesman problem.
    It is essentially a collection of City objects with auxiliary functions to
    calculate the total distance or swap two cities.
    """
    def __init__(self, cities: list[City]):
        """
        Create a new Tour.

        :param cities: (list) A list of all the City objects in this tour.
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.tour_distance = self.calculate_tour_distance()

    def calculate_tour_distance(self) -> float:
        """
        Calculates the total distance between all Cities in this Tour. This
        includes the distance from the ending city to the starting city.

        :return: (float) The distance between all cities in the tour.
        """
        # distance from city 1 to city 2, then city 2 to city 3, and so on
        # then distance from last city to first city
        total_distance = 0
        for i in range(self.num_cities - 1):
            this_distance = self.cities[i].distance_to(self.cities[i + 1])
            total_distance += this_distance

        start_end_distance = self.cities[0].distance_to(self.cities[-1])
        total_distance += start_end_distance

        return total_distance


    def swap_cities(self, position: int, shift: int) -> "Tour":
        """
        Creates a copy of this Tour with two cities swapped.
        The city at the given ``position``
        is swapped with the city that is ``shift`` units ahead of it, wrapping
        around if necessary.

        For example, with a current tour of 5 cities, ``[0, 1, 2, 3, 4]``, a
        ``position`` of `3` and a ``shift`` of `2`, index `3` would be swapped
        with index `3 + 2 mod 5`, or index `0`.

        :param position: (int) The index of the first city to swap.
        :param shift: (int) Swap positions with the city this many units ahead
            of ``position``.
        :return: (Tour) A new Tour object the same as this one, but with the two
            cities swapped.
        """

        swap_index = (position + shift) % self.num_cities  # wrap around
        cities_copy = copy.deepcopy(self.cities)
        cities_copy[position], cities_copy[swap_index] = cities_copy[
            swap_index], cities_copy[position]

        swapped_tour = Tour(cities_copy)  # should automatically update distance
        return swapped_tour


    # def generate_updated_tour(self) -> "Tour":
    #     """"""
    #     # just call swap_cities for now
    #     # want to re-calculate the tour distance after doing this


    # def draw_tour(self, include_segments=False):
    #     """"""
    #     # get a visualization of the tour and the points in it

    def draw_tour(self, include_start_end=False, show_segments=True,
                  plot_title="Tour Visualization") -> None:
        """
        Shows a visualization of the tour and the points in it.

        *Creation of this function was assisted by GitHub Copilot.*

        :param include_segments: (bool) If True, include lines connecting the
            cities.
        """
        # Extract x and y coordinates of the cities
        x_coords = [city.x for city in self.cities]
        y_coords = [city.y for city in self.cities]

        # Add the starting city to the end to close the loop
        if include_start_end:
            x_coords.append(self.cities[0].x)
            y_coords.append(self.cities[0].y)

        # Plot the cities as points
        plot.scatter(x_coords, y_coords, color='darkslateblue', label='Cities')

        # Connect the cities with lines if include_segments is True
        if show_segments:
            plot.plot(x_coords, y_coords, color='mediumseagreen', linestyle='-',
                     label='Tour Path')

        # Annotate the cities with their names
        for city in self.cities:
            plot.text(city.x, city.y, city.name, fontsize=8, ha='right')

        # Add labels and legend
        # plot.xlabel('X Coordinate')
        # plot.ylabel('Y Coordinate')
        plot.title(f"{plot_title}, distance:  {self.calculate_tour_distance():.3f}")
        # plot.legend()
        # plot.grid(True)
        plot.axis("off")
        plot.show()



    def __str__(self):
        """"""
        self_str = "Tour: "
        for city in self.cities:
            self_str += city.name + " "
        return self_str

