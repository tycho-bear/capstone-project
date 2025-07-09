# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import math
import numpy
import numpy as np
import copy
import matplotlib.pyplot as plot
import statistics
from config import (THICKNESS_MIN, THICKNESS_MAX, THICKNESS_SCALAR, RADIUS_MIN,
                    RADIUS_MAX, LENGTH_MIN, LENGTH_MAX)


class City:
    """
    Class representing a city. It has a name and two x/y or lat/lon coordinates.
    """

    def __init__(self, name: str, x: float, y: float) -> None:
        """
        Creates a new City at the given coordinates.

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

    def __eq__(self, other):
        """
        Equality checker for ordered crossover.

        :param other: The other object.
        :return: Boolean T/F
        """

        if not isinstance(other, City):
            return False
        if not other.name == self.name:
            return False
        if not other.x == self.x:
            return False
        if not other.y == self.y:
            return False

        return True


class Tour:
    """
    Class representing a tour of cities in the traveling salesman problem.
    It is essentially a collection of City objects with auxiliary functions to
    calculate the total distance or swap two cities.
    """

    def __init__(self, cities: list[City]) -> None:
        """
        Create a new Tour with the given list of Cities.

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


    def shuffle_tour(self) -> "Tour":
        """
        Helper function that shuffles the city order in this Tour. Returns a new
        shuffled Tour object.

        :return: (Tour) A new Tour object containing the cities in this Tour,
            but shuffled randomly.
        """

        self_cities = copy.deepcopy(self.cities)
        np.random.shuffle(self_cities)  # this works
        shuffled_tour = Tour(self_cities)
        return shuffled_tour


    def draw_tour(self, include_start_end=False, show_segments=True,
                  include_names=False, plot_title="Tour Visualization") -> None:
        """
        Uses matplotlib to show a visualization of the tour and its points.

        :param include_start_end: (bool) Whether to include a segment connecting
            the start and end points.
        :param show_segments: (bool) Whether to show lines between the points.
        :param plot_title: (str) The title of the plot. The tour distance should
            be included here for clarity.
        :return: None
        """

        plot.figure(figsize=(12, 10))  # make it bigger like in helper_functions

        # get coordinates
        x_coords = []
        y_coords = []
        for city in self.cities:
            x_coords.append(city.x)
            y_coords.append(city.y)

        # put the first city at the end of the list so the line connects to it
        # this closes the loop
        if include_start_end:
            x_coords.append(self.cities[0].x)
            y_coords.append(self.cities[0].y)

        plot.scatter(x_coords, y_coords, color="darkslateblue", s=140)
        if show_segments:  # include segments between the cities
            plot.plot(x_coords, y_coords, color="mediumseagreen",
                      linestyle="-", linewidth=3)

        # including city names
        if include_names:
            for city in self.cities:
                plot.text(city.x, city.y, city.name, fontsize=9, ha="right")

        plot.title(plot_title, fontsize=30)
        plot.axis("off")
        plot.tight_layout()  # does this work?
        plot.show()


    def __str__(self):
        """"""
        self_str = "Tour: "
        for city in self.cities:
            self_str += city.name + " "
        return self_str


# class Item:
#     """"""
#
#     def __init__(self):
#         """"""



class BinConfiguration:
    """
    Class representing a configuration of bins for the bin packing problem.
    It is essentially a list of item weights with auxiliary functions to
    determine the bin arrangement and fitness.

    After determining the bin arrangement, stores each bin as a list.
    """

    def __init__(self, item_weights: list[int], bin_capacity) -> None:
        """"""

        self.ITEM_WEIGHTS = item_weights
        """This and the bin capacity are what define a configuration."""
        self.BIN_CAPACITY = bin_capacity
        self.bins = self.first_fit_packing()
        """List of lists"""
        self.num_bins = len(self.bins)
        self.num_weights = len(self.ITEM_WEIGHTS)
        self.fitness = self.num_bins


    # fitness - # bins, but also penalize even distributions
    # def calculate_num_bins(self):
    #     """"""



    # first fit packing, get the bins here
    def first_fit_packing(self):
        """
        Uses the list of weights to pack items into bins. Each item is checked
        against the existing bins, and is placed in the first one where it fits.
        If it doesn't fit anywhere, a new bin is created.

        https://en.wikipedia.org/wiki/First-fit_bin_packing

        The number of bins and their distribution of items is dependent on what
        self.ITEM_WEIGHTS looks like. Swapping two weights may produce a
        different solution.

        :return:
        """

        bins = []

        for weight in self.ITEM_WEIGHTS:
            placed = False
            for bin in bins:  # check each bin
                # put it in if we can
                if (sum(bin) + weight) <= self.BIN_CAPACITY:
                    bin.append(weight)
                    placed = True
                    break

            # create a new bin here
            if not placed:
                new_bin = [weight]
                bins.append(new_bin)

        return bins




    # swap 2 bins?
    def swap_two_bins(self):
        """
        Swaps two random bins in this configuration. This may or may not affect
        the number of bins used. Returns the new configuration as a new object.

        :return:
        """

        # start, end = sorted(np.random.choice(num_cities, size=2, replace=False))
        pos1, pos2 = sorted(np.random.choice(self.num_weights, size=2,
                                             replace=False))
        new_weights = copy.deepcopy(self.ITEM_WEIGHTS)
        # do the swap
        new_weights[pos1], new_weights[pos2] = new_weights[pos2], new_weights[
            pos1]

        # create a new object
        swapped_configuration = BinConfiguration(new_weights, self.BIN_CAPACITY)
        return swapped_configuration


    # need to do something similar to shuffle_tour()
    def shuffle_bins(self):
        """
        Helper function that shuffles the weights in this configuration. Returns
        a new BinConfiguration object with the same item weights, but shuffled.

        :return: (BinConfiguration) A new BinConfiguration object with the same
            items, but shuffled.
        """

        self_weights = copy.deepcopy(self.ITEM_WEIGHTS)
        np.random.shuffle(self_weights)
        shuffled_config = BinConfiguration(self_weights, self.BIN_CAPACITY)
        return shuffled_config


    # reverse a segment  (need this for TSP too?)  (later)


    # scramble a segment  (need this for TSP too?)  (later)


    # visualize fitness (show number of bins, plus how full they are) (see pic)


class Design:
    """
    Class representing the design of a cylindrical pressure vessel.
    """

    def __init__(self,
                 head_thickness: float,
                 body_thickness: float,
                 inner_radius: float,
                 cylindrical_length: float,
                 radius_step_size: float = 0.2,
                 length_step_size: float = 0.5,
                 penalty_constant: float = 100000
                 ) -> None:
        """"""

        self.head_thickness = head_thickness
        self.body_thickness = body_thickness
        self.inner_radius = inner_radius
        self.cylindrical_length = cylindrical_length

        self.radius_step_size = radius_step_size
        self.length_step_size = length_step_size

        # then calculate total cost (see penalty method)
        self.cost = self.calculate_penalized_cost(penalty_constant)

    def get_constraints(self):
        """"""

        g1 = -1 * self.head_thickness + 0.0193 * self.inner_radius
        g2 = -1 * self.body_thickness + 0.00954 * self.inner_radius
        g3 = (-math.pi * (
                    self.inner_radius ** 2) * self.cylindrical_length -
              (4 / 3) * math.pi * (self.inner_radius ** 3) + 1296000)
        g4 = self.cylindrical_length - 240

        return [g1, g2, g3, g4]


    def is_valid_design(self):
        """"""
        # constraints = get_constraints(solution)
        constraints = self.get_constraints()
        for constraint in constraints:
            if constraint > 0:  # they all have to be <= 0, see paper
                return False

        return True


    # cost function
    def calculate_penalized_cost(self, penalty_constant:float=100000):
        """"""
        # adds a large number to the cost when constraints are violated

        base_cost = ((0.6224*self.head_thickness*self.inner_radius*
                     self.cylindrical_length)
                     + (1.7781*self.body_thickness*(self.inner_radius**2))
                     + (3.1661*(self.head_thickness**2)*self.cylindrical_length)
                     + (19.84*(self.head_thickness**2)*self.inner_radius))

        # constraints = get_constraints(self)
        constraints = self.get_constraints()
        constraint_penalty = 0

        for constraint in constraints:
            if constraint > 0:  # penalty here, > 0
                constraint_penalty += constraint**2  # quadratic penalty
                constraint_penalty += 2000  # do this to be safe
                # constraint_penalty = 20000  # just add a flat 2000

        constraint_penalty *= penalty_constant  # uncomment if doing quadratic
        return base_cost + constraint_penalty


    def generate_neighbor(self):
        """"""

        # randomly pick 1-4 variables to perturb
        # use a gaussian with step size
        # thicknesses must be discrete
        # each variable will be perturbed in a different way
        # create new variables for each one

        choices = ["head", "body", "radius", "length"]
        # randomly pick 1-3 choices to perturb
        num_choices = np.random.randint(low=1, high=4)  # never all at once
        perturbations = np.random.choice(choices, size=num_choices,
                                         replace=False)

        # apply the perturbations here, then create and return the new neighbor
        (new_head_thickness,
         new_body_thickness,
         new_inner_radius,
         new_cylindrical_length) = self.apply_perturbations(perturbations)

        new_neighbor = Design(head_thickness=new_head_thickness,
                              body_thickness=new_body_thickness,
                              inner_radius=new_inner_radius,
                              cylindrical_length=new_cylindrical_length)
        return new_neighbor


    def apply_perturbations(self, perturbations,
                            # radius_step_size:float=0.2,
                            # length_step_size:float=2
                            ):
        """"""
        # new design variables for the new neighbor, these may be modified
        new_head_thickness = copy.deepcopy(self.head_thickness)
        new_body_thickness = copy.deepcopy(self.body_thickness)
        new_inner_radius = copy.deepcopy(self.inner_radius)
        new_cylindrical_length = copy.deepcopy(self.cylindrical_length)

        # apply 1-4 perturbations
        # generate perturbation, apply it, then clip to bounds
        for perturbation in perturbations:
            if perturbation == "head":
                # can go negative or positive
                delta = THICKNESS_SCALAR * np.random.choice([-2, -1, 1, 2])
                new_head_thickness += delta
                new_head_thickness = np.clip(new_head_thickness,
                                             a_min=THICKNESS_MIN,
                                             a_max=THICKNESS_MAX)
            if perturbation == "body":
                delta = THICKNESS_SCALAR * np.random.choice([-2, -1, 1, 2])
                new_body_thickness += delta
                new_body_thickness = np.clip(new_body_thickness,
                                             a_min=THICKNESS_MIN,
                                             a_max=THICKNESS_MAX)
            if perturbation == "radius":
                delta = np.random.normal(loc=0, scale=self.radius_step_size)
                new_inner_radius += delta
                new_inner_radius = np.clip(new_inner_radius,
                                           a_min=RADIUS_MIN,
                                           a_max=RADIUS_MAX)
            if perturbation == "length":
                delta = np.random.normal(loc=0, scale=self.length_step_size)
                new_cylindrical_length += delta
                new_cylindrical_length = np.clip(new_cylindrical_length,
                                                 a_min=RADIUS_MIN,
                                                 a_max=RADIUS_MAX)

        return (new_head_thickness, new_body_thickness,
                new_inner_radius, new_cylindrical_length)






