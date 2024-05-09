import os
import numpy as np
from math import sqrt, cos, sin
from random import random, gauss
from winsound import Beep
from tkinter import *
from PIL import Image as PILImage
from PIL import ImageTk
from tkinter import Entry, Button

# Define the directory constant
DIRECTORY = os.getcwd()
# Define the filename constant
FILENAME = "test"

EPSILON = 0.0001
HUGEVALUE = 1000000.0  # 1 million
MAXDEPTH = 4  # max ray bounces
PI = 3.1415926535897932384
TWO_PI = 6.2831853071795864769
INVERTED_PI = 0.3183098861837906912

class Vector3D:
    # Initializer
    def __init__(self, x_element, y_element, z_element):
        self.x = x_element
        self.y = y_element
        self.z = z_element

    # Operator Overloading
    def __sub__(self, v):
        return Vector3D(self.x - v.x, self.y - v.y, self.z - v.z)
    def __add__(self, v):
        return Vector3D(self.x + v.x, self.y + v.y, self.z + v.z)
    def __mul__(self, s):
        return Vector3D(self.x * s, self.y * s, self.z * s)
    def __truediv__(self, s):
        return Vector3D(self.x / s, self.y / s, self.z / s)

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_numpy(v):
        return Vector3D(v[0], v[1], v[2])

# Return dot product between two vectors
def Dot(a, b):
    # Convert Vector3D to numpy arrays for calculation
    a_np = a.to_numpy() if isinstance(a, Vector3D) else a
    b_np = b.to_numpy() if isinstance(b, Vector3D) else b

    # Calculate dot product
    result = np.dot(a_np, b_np)

    return result

def Cross(a, b):
    # Convert Vector3D to numpy arrays for calculation
    a_np = a.to_numpy() if isinstance(a, Vector3D) else a
    b_np = b.to_numpy() if isinstance(b, Vector3D) else b

    # Calculate cross product
    result = np.cross(a_np, b_np)

    # Convert result back to Vector3D if inputs were Vector3D
    if isinstance(a, Vector3D) and isinstance(b, Vector3D):
        result = Vector3D.from_numpy(result)

    return result

def Length(v):
    # Convert Vector3D to numpy array for calculation
    v_np = v.to_numpy() if isinstance(v, Vector3D) else v

    # Calculate length
    result = np.linalg.norm(v_np)

    return result

def Normalize(v):
    # Convert Vector3D to numpy array for calculation
    v_np = v.to_numpy() if isinstance(v, Vector3D) else v

    # Add debugging print statement
    #print("v_np:", v_np, "Type:", type(v_np))

    # Calculate normalized vector
    result = v_np / np.linalg.norm(v_np)

    # Convert result back to Vector3D if input was Vector3D
    if isinstance(v, Vector3D):
        result = Vector3D.from_numpy(result)

    return result

# Return normal that is pointing on the side as the passed direction
def orient_normal(normal, direction):
    if Dot(normal, direction) < 0.0:
        return normal * -1.0  # flip normal
    else:
        return normal

class Ray:
    # Initializer
    def __init__(self, origin=Vector3D(0.0, 0.0, 0.0),
                 direction=Vector3D(0.0, 0.0, 0.0)):
        self.o = origin
        self.d = direction

    # Member Functions
    def get_hitpoint(self, t):
        return self.o + self.d * t

class RGBColour:
    # Initializer
    def __init__(self, red, green, blue):
        self.r = red
        self.g = green
        self.b = blue

    def __add__(self, c):
        return RGBColour(self.r + c.r, self.g + c.g, self.b + c.b)

    def __sub__(self, c):
        return RGBColour(self.r - c.r, self.g - c.g, self.b - c.b)

    def __mul__(self, s):
        return RGBColour(self.r * s, self.g * s, self.b * s)

    def __truediv__(self, s):
        return RGBColour(self.r / s, self.g / s, self.b / s)

    # this allows us to multiply by another RGBColour
    def multiply(self, c):
        return RGBColour(self.r * c.r, self.g * c.g, self.b * c.b)

    def clamp(self, minimum, maximum):
        # red
        if self.r > maximum:
            self.r = maximum
        if self.r < minimum:
            self.r = minimum
        # green
        if self.g > maximum:
            self.g = maximum
        if self.g < minimum:
            self.g = minimum
        # blue
        if self.b > maximum:
            self.b = maximum
        if self.b < minimum:
            self.b = minimum

# Constants
BLACK = RGBColour(0.0, 0.0, 0.0)
WHITE = RGBColour(1.0, 1.0, 1.0)
RED = RGBColour(1.0, 0.0, 0.0)  # for testing

def SampleHemisphere(u1, u2, exp):
    z = pow(1.0 - u1, 1.0 / (exp + 1.0))
    phi = TWO_PI * u2  # Azimuth
    theta = sqrt(max(0.0, 1.0 - z * z))  # Polar
    p = np.array([theta * cos(phi), theta * sin(phi), z])
    return Vector3D.from_numpy(p)

def OrientedHemiDir(u1, u2, normal, exp):
    p = SampleHemisphere(u1, u2, exp)  # random point on hemisphere
    # create orthonormal basis around normal
    w = normal.to_numpy()
    v = Normalize(Cross(Vector3D(0.00319, 1.0, 0.0078), Vector3D.from_numpy(w))).to_numpy()
    u = Normalize(Cross(Vector3D.from_numpy(v), Vector3D.from_numpy(w))).to_numpy()
    hemi_dir = Vector3D(u[0] * p.x + v[0] * p.y + w[0] * p.z,
                        u[1] * p.x + v[1] * p.y + w[1] * p.z,
                        u[2] * p.x + v[2] * p.y + w[2] * p.z)
    return Normalize(hemi_dir)  # normalized

class BxDF:
    def __init__(self):
        self.ke = BLACK  # default, unless set with set_emission()

    def set_emission(self, emission_colour):
        self.ke = emission_colour

    def get_emission(self):
        return self.ke

class Lambertian(BxDF):
    # Initializer
    def __init__(self, diffuse_colour):
        BxDF.__init__(self)  # call base constructor
        self.kd = diffuse_colour

    # Member Functions
    def f(self, incoming, outgoing, normal):
        return self.kd * INVERTED_PI  # Colour

    # return tuple (incoming direction, pdf)
    def sample_f(self, normal, outgoing):
        u1 = random()
        u2 = random()
        wi = OrientedHemiDir(u1, u2, normal, 0.0)

        pdf = Dot(normal, wi.to_numpy()) * INVERTED_PI
        return (wi, pdf)

class PerfectSpecular(BxDF):
    def __init__(self, specular_colour):
        BxDF.__init__(self)  # call base constructor
        self.ks = specular_colour

    # Member Functions
    def f(self, incoming, outgoing, normal):
        return self.ks  # Colour

    # return tuple (incoming direction, pdf)
    def sample_f(self, normal, outgoing):
        ndotwo = Dot(normal, outgoing)
        # perfect mirror reflection
        wi = (outgoing * -1.0) + normal * 2.0 * ndotwo
        pdf = Dot(normal, wi.to_numpy())
        return (wi, pdf)

class GlossySpecular(BxDF):
    def __init__(self, specular_colour, specular_exponent):
        BxDF.__init__(self)  # call base constructor
        self.ks = specular_colour
        self.exp = specular_exponent

    # Member Functions
    def f(self, incoming, outgoing, normal):
        ndotwi = Dot(normal, incoming)  # normal and incoming light
        reflect_dir = (incoming * -1.0) + normal * 2.0 * ndotwi
        rdotwo = Dot(reflect_dir.to_numpy(), outgoing.to_numpy())
        if rdotwo > 0.0:
            return self.ks * pow(rdotwo, self.exp)
        else:
            return BLACK

    def sample_f(self, normal, outgoing):
        # perfect mirror reflection
        ndotwo = Dot(normal, outgoing)
        reflect_dir = (outgoing * -1.0) + normal * 2.0 * ndotwo
        # orthonormal basis
        w = reflect_dir.to_numpy()
        v = Normalize(Cross(Vector3D(0.00419, 1.0, 0.0078), Vector3D.from_numpy(w))).to_numpy()
        u = Normalize(Cross(Vector3D.from_numpy(v), Vector3D.from_numpy(w))).to_numpy()
        # random samples
        u1 = random()  # [0, 1]
        u2 = random()  # [0, 1]
        p = SampleHemisphere(u1, u2, self.exp).to_numpy()  # point in hemi
        wi = (u * p[0]) + (v * p[1]) + (w * p[2])  # linear projection
        if Dot(normal, Vector3D.from_numpy(wi).to_numpy()) < 0.0:  # if reflected direction is below surface
            wi = (u * -p[0]) + (v * -p[1]) + (w * p[2])  # reflect it
        # phong lobe
        phong_lobe = pow(Dot(reflect_dir, wi), self.exp)
        pdf = Dot(normal, Vector3D.from_numpy(wi).to_numpy()) * phong_lobe
        return (Vector3D.from_numpy(wi), pdf)


class Primitive:
    # Setters
    def set_BxDF(self, BxDF):
        self.BxDF = BxDF

    # Getters
    def get_BxDF(self):
        return self.BxDF

class Sphere(Primitive):
    # Initializer
    def __init__(self, sphere_origin, sphere_radius):
        self.origin = sphere_origin
        self.radius = sphere_radius
        self.radius_squared = sphere_radius * sphere_radius  # optimization

    def intersect(self, ray):
        ray_dir = Normalize(ray.d)
        temp = ray.o - self.origin
        A = Dot(ray_dir, ray_dir)
        B = 2.0 * Dot(ray_dir, temp)
        C = Dot(temp, temp) - self.radius_squared
        disc = (B * B) - (4.0 * A * C)  # Discriminant
        if disc < 0.0:  # No Hit
            hit = False
            distance = 0.0
            hit_point = Vector3D(0.0, 0.0, 0.0)
            normal = Vector3D(0.0, 0.0, 0.0)
            return (hit, distance, hit_point, normal)  # tuple
        sqrt_disc = sqrt(disc)  # square root of discriminant
        tmin = (-B - sqrt_disc) / (2.0 * A)
        if tmin >= EPSILON:  # Hit
            hit = True
            distance = tmin
            hit_point = ray.get_hitpoint(tmin)
            normal = Normalize((hit_point - self.origin) / self.radius)
            return (hit, distance, hit_point, normal)  # tuple
        tmax = (-B + sqrt_disc) / (2.0 * A)
        if tmax >= EPSILON:  # Hit
            hit = True
            distance = tmax
            hit_point = ray.get_hitpoint(tmax)
            normal = Normalize((hit_point - self.origin) / self.radius)
            return (hit, distance, hit_point, normal)  # tuple
        hit = False
        hit_point = Vector3D(0.0, 0.0, 0.0)
        distance = 0.0
        normal = Vector3D(0.0, 0.0, 0.0)
        return (hit, distance, hit_point, normal)  # tuple

class Plane(Primitive):
    # Initializer
    def __init__(self, plane_origin, plane_normal):
        self.origin = plane_origin
        self.normal = Normalize(plane_normal)

    def intersect(self, ray):
        ray_dir = Normalize(ray.d)
        denominator = Dot(ray_dir, self.normal)
        if denominator == 0.0:  # Check for division by zero
            # ray is parallel, no hit
            hit = False
            distance = 0.0
            hit_point = Vector3D(0.0, 0.0, 0.0)
            return (hit, distance, hit_point, self.normal)  # tuple
        t = Dot(self.normal, (self.origin - ray.o)) / denominator
        if t >= EPSILON:  # Hit
            hit = True
            distance = t
            hit_point = ray.get_hitpoint(t)
            return (hit, distance, hit_point, self.normal)  # tuple
        hit = False
        distance = 0.0
        hit_point = Vector3D(0.0, 0.0, 0.0)
        return (hit, distance, hit_point, self.normal)  # tuple

class PathTraceIntegrator:
    # Initializer - creates object list
    def __init__(self):
        self.primitives = []

    # trace light path
    def trace_ray(self, ray, depth):
        result = RGBColour(0.0, 0.0, 0.0)  # black
        t = HUGEVALUE
        index = -1  # -1 means no hit
        if depth > MAXDEPTH:
            return result
        for i in range(0, len(self.primitives)):
            # intersect returns tuple of (bool hit, distance, hit_point, normal)
            hit_data = self.primitives[i].intersect(ray)
            if hit_data[0] == True:  # Hit
                if hit_data[1] < t:  # Distance
                    t = hit_data[1]
                    hit_point = hit_data[2]  # hit_point
                    normal = hit_data[3]  # normal
                    index = i  # closest primitive index number
        if index == -1:  # No Hit
            return BLACK
        else:  # Hit
            wo = ray.d * -1.0  # outgoing (towards camera)
            normal = orient_normal(normal, wo)  # make normal point in correct direction
            shading_data = self.primitives[index].get_BxDF().sample_f(normal, wo)
            wi = shading_data[0]  # incoming direction
            pdf = shading_data[1]  # pdf
            if pdf <= 0.0:
                pdf = 1.0
            f = self.primitives[index].get_BxDF().f(wi, wo, normal)
            incoming_ray = Ray(hit_point, wi)  # make incoming to follow
            RR_prob = 0.66
            if depth > 2:
                if random() < RR_prob:  # 2/3 chance we stop here
                    return result
            result = result + f.multiply(self.trace_ray(incoming_ray, depth + 1)) * Dot(wi.to_numpy(), normal.to_numpy()) / pdf
            # Add emission
            result = result + self.primitives[index].get_BxDF().get_emission()
            result = result / RR_prob
        return result  # return final colour

    def add_primitive(self, primitive):
        self.primitives.append(primitive)

class Camera:
    # Initializer
    def __init__(self, eye_point, focal_point, view_distance, up_vector,
                 image_height, image_width, samples_per_pixel):
        self.eye = eye_point
        self.focal = focal_point
        self.view_dist = view_distance
        self.up = up_vector
        self.height = image_height
        self.width = image_width
        self.spp = samples_per_pixel
        self.compute_uvw()
        # create empty image array
        self.image_array = np.zeros((image_width, image_height, 3), dtype=np.uint8)

    def compute_uvw(self):
        # w
        self.w = self.eye - self.focal
        self.w = Normalize(self.w)
        # u
        self.u = Cross(self.up, self.w)
        self.u = Normalize(self.u)
        # v
        self.v = Cross(self.w, self.u)
        self.v = Normalize(self.v)

        if (self.eye.x == self.focal.x and
            self.eye.z == self.focal.z and
            self.focal.y < self.eye.y):
            self.u = Vector3D(0.0, 0.0, 1.0)
            self.v = Vector3D(1.0, 0.0, 0.0)
            self.w = Vector3D(0.0, 1.0, 0.0)

        if (self.eye.x == self.focal.x and
            self.eye.z == self.focal.z and
            self.focal.y > self.eye.y):
            self.u = Vector3D(1.0, 0.0, 0.0)
            self.v = Vector3D(0.0, 0.0, 1.0)
            self.w = Vector3D(0.0, -1.0, 0.0)

    def get_direction(self, x, y):
        direction = (self.u * x) + (self.v * y) - (self.w * self.view_dist)
        return Normalize(direction)

    #def save_image(self, filename):
        # Save image using Pillow
        #image = PILImage.fromarray(self.image_array, mode="RGB")
        #image.save(os.path.join(DIRECTORY, filename))
        #print("Image Saved")

        # Play sound to signal a beep (For Windows)
        #for i in range(1, 4):
        #    Beep(i * 500, 250)
    
    def save_image(self, filename):
        # Save image using Pillow
        image = PILImage.fromarray(self.image_array, mode="RGB")
        # Flip the image vertically
        image = image.transpose(PILImage.FLIP_TOP_BOTTOM)
        # Append .jpg to the filename
        image.save(os.path.join(DIRECTORY, filename + ".jpg"))
        print("Image Saved")

        # Play sound to signal a beep (For Windows)
        for i in range(1, 4):
            Beep(i * 500, 250)

    def render(self, integrator):
        ray = Ray()
        ray.o = self.eye
        pixel = BLACK  # create black pixel
        for x in range(0, self.width):
            for y in range(0, self.height):
                pixel = BLACK  # start at black
                for s in range(0, self.spp):
                    sp_x = (x + random()) - (self.width / 2.0)
                    sp_y = (y + random()) - (self.height / 2.0)
                    ray.d = self.get_direction(sp_x, sp_y)
                    pixel = pixel + integrator.trace_ray(ray, 1)
                pixel = pixel / self.spp
                # Clamp pixel values to range 0-255
                pixel.r = min(max(pixel.r, 0), 1)
                pixel.g = min(max(pixel.g, 0), 1)
                pixel.b = min(max(pixel.b, 0), 1)

                # Convert pixel values to integers
                self.image_array[y, x] = [int(pixel.r * 255), int(pixel.g * 255), int(pixel.b * 255)]
            print((x / self.width) * 100, "%")

        # Save the rendered image
        self.save_image(FILENAME)


def render_image():
    # Get the spp value from the input field
    spp = int(spp_entry.get())
    
    # Set the spp value in your Camera object
    cam.spp = spp
    
    # Render the image
    cam.render(path_tracer)
    
    # Save the image
    cam.save_image(FILENAME)
    
    # Update the image in the viewer
    image_name = ImageTk.PhotoImage(PILImage.open(os.path.join(DIRECTORY, FILENAME + ".jpg")))
    viewer.itemconfig(image_item, image=image_name)


path_tracer = PathTraceIntegrator()
gold_diff = Lambertian(RGBColour(1.0, 0.8, 0.3))
ground_diff = Lambertian(RGBColour(0.15, 0.15, 0.15))
red_emitt = Lambertian(RGBColour(3.0, 0.0, 0.0))
red_emitt.set_emission(RGBColour(3.0, 0.0, 0.0))
blue_emitt = Lambertian(RGBColour(0.0, 0.0, 3.0))
blue_emitt.set_emission(RGBColour(0.0, 0.0, 3.0))
grey_emitt_plane = Lambertian(RGBColour(0.2, 0.2, 0.2))
grey_emitt_plane.set_emission(RGBColour(0.2, 0.2, 0.2))
mirror = PerfectSpecular(RGBColour(1.0, 1.0, 1.0))
glossy = GlossySpecular(RGBColour(0.2, 1.0, 0.3), 35.0)
# sphere 1 - yellow main
sphere_1 = Sphere(Vector3D(0.0, 0.0, 0.0), 18.0)
sphere_1.set_BxDF(gold_diff)
path_tracer.add_primitive(sphere_1)
# sphere 2 - red sphere light
sphere_2 = Sphere(Vector3D(-20.0, 22.0, 10.0), 6.25)
sphere_2.set_BxDF(red_emitt)
path_tracer.add_primitive(sphere_2)
# sphere 3 - blue sphere light on ground
sphere_3 = Sphere(Vector3D(20.0, -13.0, 25.0), 4.0)
sphere_3.set_BxDF(blue_emitt)
path_tracer.add_primitive(sphere_3)
# sphere 4 - mirror front
sphere_4 = Sphere(Vector3D(4.0, -8.0, 20.0), 8.0)
sphere_4.set_BxDF(glossy)
path_tracer.add_primitive(sphere_4)
# plane 1 - bottom ground
plane_1 = Plane(Vector3D(0.0, -16.0, 0.0), Vector3D(0.0, 1.0, 0.0))
plane_1.set_BxDF(ground_diff)
path_tracer.add_primitive(plane_1)
# plane 2 - top light
plane_2 = Plane(Vector3D(0.0, 45.0, 0.0), Vector3D(0.0, -1.0, 0.0))
plane_2.set_BxDF(grey_emitt_plane)
path_tracer.add_primitive(plane_2)
# Create Camera
eye = Vector3D(-3.0, 0.0, 190.0)  # higher z = more narrow view
focal = Vector3D(0.0, 0.0, 0.0)
view_distance = 1000  # larger = more orthographic like
up = Vector3D(0.0, 1.0, 0.0)
height = 400
width = 400
spp = 1
cam = Camera(eye, focal, view_distance, up, height, width, spp)
cam.render(path_tracer)  # trace scene and save image

# Create the Tkinter window
root = Tk()
root.title("PyPath")

# Create the viewer
viewer = Canvas(root, width=width, height=height)
image_name = ImageTk.PhotoImage(PILImage.open(os.path.join(DIRECTORY, FILENAME + ".jpg")))
image_item = viewer.create_image(width/2.0, height/2.0, image=image_name)
viewer.grid(row=0, column=0)

# Create the spp input field
spp_entry = Entry(root)
spp_entry.insert(0, str(cam.spp))  # Set the initial value to the current spp
spp_entry.grid(row=1, column=0)

# Create the render button
render_button = Button(root, text="Render", command=render_image)
render_button.grid(row=2, column=0)

root.mainloop()
