"""Rigid crumpled-paper visuals with one cheap box collision approximation each."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull


def install_crumpled_tissues(root):
    world = root.find('worldbody')
    assets = root.find('asset')
    if assets is None:
        assets = ET.SubElement(root, 'asset')
    for number, name in enumerate(('tissue_1', 'tissue_2')):
        body = world.find(f"body[@name='{name}']")
        if body is None:
            raise ValueError(f'Missing {name}')
        for g in list(body.findall('geom')):
            body.remove(g)
        rng = np.random.default_rng(210 + number)
        count = 162
        i = np.arange(count)
        z = 1 - 2*(i+.5)/count
        angle = i*np.pi*(3-np.sqrt(5))
        points = np.column_stack((np.sqrt(1-z*z)*np.cos(angle),
                                  np.sqrt(1-z*z)*np.sin(angle), z))
        faces = ConvexHull(points).simplices.copy()
        # Irregular inward folds. No thin collision facets are created.
        radius = .012 * rng.uniform(.80, 1.10, count)
        vertices = points * radius[:, None] * [1., 1., .87]
        for face in faces:
            a, b, c = vertices[face]
            if np.dot(np.cross(b-a,c-a), (a+b+c)/3) < 0:
                face[1], face[2] = face[2], face[1]
        mesh_name = name + '_crumpled_visual_mesh'
        for old in list(assets.findall(f"mesh[@name='{mesh_name}']")):
            assets.remove(old)
        # Explicit face normals retain the creases rather than smoothing them.
        normals = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],
                           vertices[faces[:,2]]-vertices[faces[:,0]])
        normals /= np.linalg.norm(normals, axis=1)[:,None]
        def flat(values):
            return ' '.join(str(x) for x in np.asarray(values).ravel())
        ET.SubElement(assets, 'mesh', name=mesh_name,
            vertex=flat(vertices[faces].reshape(-1,3)),
            face=flat(np.arange(len(faces)*3).reshape(-1,3)),
            normal=flat(np.repeat(normals,3,axis=0)))
        ET.SubElement(body, 'geom', name=name+'_collision', type='box',
            size='.009 .009 .009', rgba='0 0 0 0', mass='.001',
            friction='.8 .01 .001', solref='.004 1', solimp='.95 .99 .001', condim='4')
        ET.SubElement(body, 'geom', name=name+'_paper_visual', type='mesh', mesh=mesh_name,
            rgba='.96 .95 .92 1', contype='0', conaffinity='0', mass='0', group='2')
        xyz = np.fromstring(body.get('pos'), sep=' ')
        xyz[2] = .0135
        body.set('pos', flat(xyz))
        site = body.find(f"site[@name='{name}_grasp']")
        if site is None:
            raise ValueError(f'Missing grasp site for {name}')
        site.set('pos', '0 0 .004')
