import numpy
from scipy.spatial import Voronoi, distance
from scipy.stats import gaussian_kde
from noise import pnoise2, pnoise3
from uuid import uuid4 as uuid
from time import time
import svgwrite
import cairosvg
from colour import Color


class Terrain:
    def __init__(self, n=10, h=1000, w=1000):
        self.id = uuid()
        self.height = h
        self.width = w

        self.nodes = numpy.empty([0, 2])
        self.edges = numpy.empty([0, 2])
        self.faces = numpy.empty([0, 3])

        self.elevations = numpy.empty([0, 1])

        self.generate_terrain(2 ** n)

    def nodes_2d(self):
        return self.nodes

    def nodes_3d(self):
        return numpy.column_stack([self.nodes, self.elevations])

    def edge_nodes(self):
        for a, b in self.edges:
            node_a = self.nodes[a]
            node_b = self.nodes[b]
            yield node_a, node_b

    def face_nodes(self):
        for a, b, c in self.faces:
            node_a = self.nodes[a]
            node_b = self.nodes[b]
            node_c = self.nodes[c]
            yield node_a, node_b, node_c

    def face_nodes_3d(self):
        for a, b, c in self.faces:
            node_a = self.nodes_3d()[a]
            node_b = self.nodes_3d()[b]
            node_c = self.nodes_3d()[c]
            yield node_a, node_b, node_c

    def generate_terrain(self, num_points):
        self.nodes = numpy.random.random((num_points, 2))
        self.nodes = numpy.multiply(self.nodes, [self.width, self.height])
        self.improve_nodes(3)

        self.generate_mesh()
        self.generate_elevations()

    def improve_nodes(self, iterations=1):
        voronoi = Voronoi(self.nodes)
        nodes = list()
        for point, region in zip(voronoi.points, voronoi.point_region):
            if -1 in voronoi.regions[region]:
                # If region touches edge, leave point where it is
                nodes.append(point)

            else:
                # Else move the point to the centre of the Voronoi region
                vertices = numpy.asarray([voronoi.vertices[i, :] for i in voronoi.regions[region]])
                vertices[vertices < 0] = 0
                vertices = numpy.minimum(vertices, numpy.asarray([self.width, self.height]))
                centre = vertices.mean(0)
                nodes.append(centre)

        self.nodes = numpy.asarray(nodes)

        iterations -= 1
        if iterations > 0:
            self.improve_nodes(iterations)

    def order_nodes(self):
        indexes = numpy.argsort((self.nodes.sum(1)))
        self.nodes = self.nodes.take(indexes, 0)

    def generate_mesh(self):
        self.order_nodes()

        voronoi = Voronoi(self.nodes)
        edges = list()
        faces = dict()

        for points, vertices in zip(voronoi.ridge_points, voronoi.ridge_vertices):
            if -1 not in vertices:
                inside = True
                p0, p1 = points

                for v in vertices:
                    x, y = voronoi.vertices[v]
                    if 0 < x < self.width and 0 < y < self.height:
                        faces.setdefault(v, set())
                        faces[v].add(p0)
                        faces[v].add(p1)
                    else:
                        inside = False

                if inside:
                    edges.append(points)

        self.edges = numpy.asarray(edges)
        self.faces = numpy.asarray([list(face) for face in faces.values() if len(face) == 3])

    def generate_elevations(self):
        nodes = self.nodes.T

        height_map = gaussian_kde(nodes)
        elevations = height_map(nodes)
        self.elevations = elevations

        self.scale_elevations()
        # self.add_speckle()

    def scale_elevations(self):

        elevations = self.elevations - numpy.min(self.elevations)
        elevations /= numpy.max(elevations)

        self.elevations = elevations

    def add_speckle(self, n=1.0):
        elevations = list()
        for x, y, z in self.nodes_3d():
            z += n*pnoise2(x, y)
            elevations.append(z)

        self.elevations = numpy.asarray(elevations)
        self.scale_elevations()

    def draw(self):
        filename = 'maps\{0}-{1}.svg'.format(self.id, time())

        svg = svgwrite.Drawing(filename, size=(self.width, self.height))
        for node in self.nodes_3d():
            fill = Color(hue=0, saturation=0, luminance=node[2])
            element = svgwrite.shapes.Circle(node[:2], 2, fill=fill.hex)
            svg.add(element)

        # for a, b in self.edges:
        #     node_a = self.nodes[a]
        #     node_b = self.nodes[b]
        #     element = svgwrite.shapes.Line(node_a, node_b, stroke='black')
        #     svg.add(element)

        for node_a, node_b, node_c in self.face_nodes_3d():
            height = numpy.average([node_a[2], node_b[2], node_c[2]])
            fill = Color(hue=0, saturation=0, luminance=height)
            element = svgwrite.shapes.Polygon([node_a[:2], node_b[:2], node_c[:2]], fill=fill.hex)
            svg.add(element)

        svg.save(True)

        cairosvg.svg2png(svg.tostring(), write_to=filename.replace('.svg', '.png'))


if __name__ == '__main__':
    t = Terrain(12)
    t.draw()
