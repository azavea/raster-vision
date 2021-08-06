from typing import Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon, LinearRing, MultiPolygon
from shapely.ops import unary_union
from triangle import triangulate


class AoiSampler():
    """Given a set of polygons representing the AOI, allows efficiently
    sampling points inside the AOI uniformly at random.

    To achieve this, each polygon is first partitioned into triangles
    (triangulation). Then, to sample a single point, we first sample a triangle
    at random with probability proportional to its area and then sample a point
    within that triangle uniformly at random.
    """

    def __init__(self, polygons: Sequence[Polygon]) -> None:
        # merge overlapping polygons, if any
        merged_polygons = unary_union(polygons)
        if isinstance(merged_polygons, Polygon):
            merged_polygons = [merged_polygons]
        self.polygons = MultiPolygon(merged_polygons)
        self.triangulate(self.polygons)

    def triangulate(self, polygons) -> dict:
        triangulations = [self.triangulate_polygon(p) for p in polygons]
        self.triangulations = triangulations
        self.origins = np.vstack([t['origins'] for t in triangulations])
        self.vec_AB = np.vstack([t['bases'][0] for t in triangulations])
        self.vec_AC = np.vstack([t['bases'][1] for t in triangulations])
        areas = np.concatenate([t['areas'] for t in triangulations])
        self.weights = areas / areas.sum()
        self.ntriangles = len(self.origins)

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample a random point within the AOI, using the following algorithm:
            - Randomly sample one triangle (ABC) with probability proportional
            to its area.
            - Starting at A, travel a random distance along vectors AB and AC.
            - Return the final position.

        Args:
            n (int, optional): Number of points to sample. Defaults to 1.

        Returns:
            np.ndarray: (n, 2) 2D coordinates of the sampled points.
        """
        tri_idx = np.random.choice(self.ntriangles, p=self.weights, size=n)
        origin = self.origins[tri_idx]
        vec_AB = self.vec_AB[tri_idx]
        vec_AC = self.vec_AC[tri_idx]
        # the fractions to travel along each of the two vectors
        r, s = np.random.uniform(size=(2, n, 1))
        # If the fractions will land us in the wrong half of the parallelogram
        # defined by vec AB and vec AC, reflect them into the correct half.
        mask = (r + s) > 1
        r[mask] = 1 - r[mask]
        s[mask] = 1 - s[mask]
        loc = origin + (r * vec_AB + s * vec_AC)
        return loc

    def triangulate_polygon(self, polygon: Polygon) -> dict:
        """Extract vertices and edges from the polygon (and its holes, if any)
        and pass them to the Triangle library for triangulation.
        """
        vertices, edges = self.polygon_to_graph(polygon.exterior)

        holes = polygon.interiors
        if not holes:
            args = {
                'vertices': vertices,
                'segments': edges,
            }
        else:
            for hole in holes:
                hole_vertices, hole_edges = self.polygon_to_graph(hole)
                # make the indices point to entries in the global vertex list
                hole_edges += len(vertices)
                # append hole vertices to the global vertex list
                vertices = np.vstack([vertices, hole_vertices])
                edges = np.vstack([edges, hole_edges])

            # the triangulation algorithm requires a sample point inside each
            # hole
            hole_centroids = np.stack([hole.centroid for hole in holes])

            args = {
                'vertices': vertices,
                'segments': edges,
                'holes': hole_centroids
            }

        tri = triangulate(args, opts='p')
        simplices = tri['triangles']
        vertices = np.array(tri['vertices'])
        origins, bases = self.triangle_origin_and_basis(vertices, simplices)

        out = {
            'vertices': vertices,
            'simplices': simplices,
            'origins': origins,
            'bases': bases,
            'areas': self.triangle_area(vertices, simplices)
        }
        return out

    def polygon_to_graph(self,
                         polygon: LinearRing) -> Tuple[np.ndarray, np.ndarray]:
        """Given the exterior of a polygon, return its graph representation.

        Args:
            polygon (LinearRing): The exterior of a polygon.

        Returns:
            Tuple[np.ndarray, np.ndarray]: An (N, 2) array of vertices and
            an (N, 2) array of indices to vertices representing edges.
        """
        vertices = np.array(polygon)
        # Discard the last vertex - it is a duplicate of the first vertex and
        # duplicates cause problems for the Triangle library.
        vertices = vertices[:-1]

        N = len(vertices)
        # Tuples of indices to vertices representing edges.
        # mod N ensures edge from last vertex to first vertex by making the
        # last tuple [N-1, 0].
        edges = np.column_stack([np.arange(0, N), np.arange(1, N + 1)]) % N

        return vertices, edges

    def triangle_side_lengths(self, vertices: np.ndarray, simplices: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate lengths of all 3 sides of each triangle specified by the
        simplices array.

        Args:
            vertices (np.ndarray): (N, 2) array of vertex coords in 2D.
            simplices (np.ndarray): (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: ||AB||, ||BC||, ||AC||
        """
        A = vertices[simplices[:, 0]]
        B = vertices[simplices[:, 1]]
        C = vertices[simplices[:, 2]]
        AB, AC, BC = B - A, C - A, C - B
        ab = np.linalg.norm(AB, axis=1)
        bc = np.linalg.norm(BC, axis=1)
        ac = np.linalg.norm(AC, axis=1)
        return ab, bc, ac

    def triangle_origin_and_basis(
            self, vertices: np.ndarray, simplices: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """For each triangle ABC, return point A, vector AB, and vector AC.

        Args:
            vertices (np.ndarray): (N, 2) array of vertex coords in 2D.
            simplices (np.ndarray): (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 3 arrays of shape
                (N, 2), organized into tuples like so:
                (point A, (vector AB, vector AC)).
        """
        A = vertices[simplices[:, 0]]
        B = vertices[simplices[:, 1]]
        C = vertices[simplices[:, 2]]
        AB = B - A
        AC = C - A
        return A, (AB, AC)

    def triangle_area(self, vertices: np.ndarray,
                      simplices: np.ndarray) -> np.ndarray:
        """Calculate area of each triangle specified by the simplices array
        using Heron's formula.

        Args:
            vertices (np.ndarray): (N, 2) array of vertex coords in 2D.
            simplices (np.ndarray): (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            np.ndarray: (N,) array of areas
        """
        a, b, c = self.triangle_side_lengths(vertices, simplices)
        p = (a + b + c) * 0.5
        area = p * (p - a) * (p - b) * (p - c)
        area[area < 0] = 0
        area = np.sqrt(area)
        return area
