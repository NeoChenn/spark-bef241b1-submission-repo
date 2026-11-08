import warp as wp

wp.config.quiet = True


@wp.func
def clampi(i: int, n: int) -> int:
    return wp.max(0, wp.min(i, n - 1))


@wp.func
def interp(f: wp.array3d(dtype=float), x: float, y: float, z: float) -> float:
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    x = wp.clamp(x, 0.0, float(nx - 1))
    y = wp.clamp(y, 0.0, float(ny - 1))
    z = wp.clamp(z, 0.0, float(nz - 1))
    i = wp.min(int(x), nx - 2)
    j = wp.min(int(y), ny - 2)
    k = wp.min(int(z), nz - 2)
    fx = x - float(i)
    fy = y - float(j)
    fz = z - float(k)
    c00 = f[i, j, k] * (1.0 - fx) + f[i + 1, j, k] * fx
    c10 = f[i, j + 1, k] * (1.0 - fx) + f[i + 1, j + 1, k] * fx
    c01 = f[i, j, k + 1] * (1.0 - fx) + f[i + 1, j, k + 1] * fx
    c11 = f[i, j + 1, k + 1] * (1.0 - fx) + f[i + 1, j + 1, k + 1] * fx
    return (c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz) + (c01 * (1.0 - fy) + c11 * fy) * fz


@wp.func
def vel(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), p: wp.vec3) -> wp.vec3:
    return wp.vec3(interp(u, p[0], p[1] - 0.5, p[2] - 0.5), interp(v, p[0] - 0.5, p[1], p[2] - 0.5),
                   interp(w, p[0] - 0.5, p[1] - 0.5, p[2]))


@wp.func
def face_pos(i: int, j: int, k: int, axis: int) -> wp.vec3:
    if axis == 0:
        return wp.vec3(float(i), float(j) + 0.5, float(k) + 0.5)
    if axis == 1:
        return wp.vec3(float(i) + 0.5, float(j), float(k) + 0.5)
    return wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k))


@wp.func
def solid_at(solid: wp.array3d(dtype=wp.int32), i: int, j: int, k: int) -> int:
    if i < 0 or j < 0 or k < 0 or i >= solid.shape[0] or j >= solid.shape[1] or k >= solid.shape[2]:
        return 0
    return solid[i, j, k]


@wp.kernel
def advect_vel(f: wp.array3d(dtype=float), u: wp.array3d(dtype=float), v: wp.array3d(dtype=float),
               w: wp.array3d(dtype=float), out: wp.array3d(dtype=float), axis: int, h: float, wind: wp.vec3,
               n: wp.vec3):
    i, j, k = wp.tid()
    p = face_pos(i, j, k, axis)
    mid = p - 0.5 * h * vel(u, v, w, p)
    q = p - h * vel(u, v, w, mid)
    if q[0] < 0.0 or q[1] < 0.0 or q[2] < 0.0 or q[0] > n[0] or q[1] > n[1] or q[2] > n[2]:
        out[i, j, k] = wind[axis]
        return
    if axis == 0:
        out[i, j, k] = interp(f, q[0], q[1] - 0.5, q[2] - 0.5)
    elif axis == 1:
        out[i, j, k] = interp(f, q[0] - 0.5, q[1], q[2] - 0.5)
    else:
        out[i, j, k] = interp(f, q[0] - 0.5, q[1] - 0.5, q[2])


@wp.kernel
def body_forces(f: wp.array3d(dtype=float), c: wp.array3d(dtype=float), solid: wp.array3d(dtype=wp.int32),
                axis: int, dt: float, buoy: float, wind: wp.vec3, n: wp.vec3, sponge: float, relax: float,
                cap: float):
    i, j, k = wp.tid()
    x = f[i, j, k]
    if axis == 2 and k > 0 and k < c.shape[2]:
        x += dt * buoy * 0.5 * (c[i, j, k - 1] + c[i, j, k])
    p = face_pos(i, j, k, axis)
    d = wp.min(wp.min(p[0], n[0] - p[0]), wp.min(p[1], n[1] - p[1]))
    a = wp.clamp(1.0 - d / sponge, 0.0, 1.0) * relax
    x += (wind[axis] - x) * a
    x = wp.clamp(x, -cap, cap)
    li = i
    lj = j
    lk = k
    if axis == 0:
        li = i - 1
    if axis == 1:
        lj = j - 1
    if axis == 2:
        lk = k - 1
    if solid_at(solid, i, j, k) != 0 or solid_at(solid, li, lj, lk) != 0:
        x = 0.0
    f[i, j, k] = x


@wp.kernel
def add_jet(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), cell: wp.vec3i,
            acc: wp.vec3, dt: float):
    i = cell[0]
    j = cell[1]
    k = cell[2]
    h = 0.5 * dt
    u[i, j, k] = u[i, j, k] + h * acc[0]
    u[i + 1, j, k] = u[i + 1, j, k] + h * acc[0]
    v[i, j, k] = v[i, j, k] + h * acc[1]
    v[i, j + 1, k] = v[i, j + 1, k] + h * acc[1]
    w[i, j, k] = w[i, j, k] + h * acc[2]
    w[i, j, k + 1] = w[i, j, k + 1] + h * acc[2]


@wp.kernel
def divergence(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
               solid: wp.array3d(dtype=wp.int32), div: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        div[i, j, k] = 0.0
        return
    div[i, j, k] = u[i + 1, j, k] - u[i, j, k] + v[i, j + 1, k] - v[i, j, k] + w[i, j, k + 1] - w[i, j, k]


@wp.func
def nb(phi: wp.array3d(dtype=float), solid: wp.array3d(dtype=wp.int32), i: int, j: int, k: int) -> wp.vec2:
    if i < 0 or j < 0 or k < 0 or i >= phi.shape[0] or j >= phi.shape[1] or k >= phi.shape[2]:
        return wp.vec2(0.0, 1.0)
    if solid[i, j, k] != 0:
        return wp.vec2(0.0, 0.0)
    return wp.vec2(phi[i, j, k], 1.0)


@wp.kernel
def sor(phi: wp.array3d(dtype=float), div: wp.array3d(dtype=float), solid: wp.array3d(dtype=wp.int32),
        color: int, omega: float):
    i, j, k = wp.tid()
    if (i + j + k) % 2 != color or solid[i, j, k] != 0:
        return
    s = nb(phi, solid, i - 1, j, k) + nb(phi, solid, i + 1, j, k) + nb(phi, solid, i, j - 1, k)
    s += nb(phi, solid, i, j + 1, k) + nb(phi, solid, i, j, k - 1) + nb(phi, solid, i, j, k + 1)
    if s[1] == 0.0:
        return
    g = (s[0] - div[i, j, k]) / s[1]
    phi[i, j, k] = (1.0 - omega) * phi[i, j, k] + omega * g


@wp.func
def phi_at(phi: wp.array3d(dtype=float), i: int, j: int, k: int) -> float:
    if i < 0 or j < 0 or k < 0 or i >= phi.shape[0] or j >= phi.shape[1] or k >= phi.shape[2]:
        return 0.0
    return phi[i, j, k]


@wp.kernel
def project(f: wp.array3d(dtype=float), phi: wp.array3d(dtype=float), solid: wp.array3d(dtype=wp.int32), axis: int):
    i, j, k = wp.tid()
    li = i
    lj = j
    lk = k
    if axis == 0:
        li = i - 1
    if axis == 1:
        lj = j - 1
    if axis == 2:
        lk = k - 1
    if solid_at(solid, i, j, k) != 0 or solid_at(solid, li, lj, lk) != 0:
        f[i, j, k] = 0.0
        return
    f[i, j, k] = f[i, j, k] - (phi_at(phi, i, j, k) - phi_at(phi, li, lj, lk))


@wp.func
def cget(c: wp.array3d(dtype=float), i: int, j: int, k: int) -> float:
    return c[clampi(i, c.shape[0]), clampi(j, c.shape[1]), clampi(k, c.shape[2])]


@wp.func
def slope(a: float, b: float) -> float:
    if a * b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


@wp.func
def flux(c: wp.array3d(dtype=float), i: int, j: int, k: int, d: wp.vec3i, uf: float, n: int, r: int) -> float:
    if uf > 0.0:
        if r == 0:
            return 0.0
        cl = cget(c, i - d[0], j - d[1], k - d[2])
        cll = cget(c, i - 2 * d[0], j - 2 * d[1], k - 2 * d[2])
        cr = cget(c, i, j, k)
        return uf * (cl + 0.5 * slope(cl - cll, cr - cl))
    if r == n:
        return 0.0
    cr = cget(c, i, j, k)
    crr = cget(c, i + d[0], j + d[1], k + d[2])
    cl = cget(c, i - d[0], j - d[1], k - d[2])
    return uf * (cr - 0.5 * slope(cr - cl, crr - cr))


@wp.func
def diff(c: wp.array3d(dtype=float), solid: wp.array3d(dtype=wp.int32), i: int, j: int, k: int, c0: float) -> float:
    if i < 0 or j < 0 or k < 0 or i >= c.shape[0] or j >= c.shape[1] or k >= c.shape[2]:
        return -c0
    if solid[i, j, k] != 0:
        return 0.0
    return c[i, j, k] - c0


@wp.kernel
def transport(c: wp.array3d(dtype=float), out: wp.array3d(dtype=float), u: wp.array3d(dtype=float),
              v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), solid: wp.array3d(dtype=wp.int32),
              h: float, kd: float):
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        out[i, j, k] = 0.0
        return
    ex = wp.vec3i(1, 0, 0)
    ey = wp.vec3i(0, 1, 0)
    ez = wp.vec3i(0, 0, 1)
    nx = c.shape[0]
    ny = c.shape[1]
    nz = c.shape[2]
    net = flux(c, i + 1, j, k, ex, u[i + 1, j, k], nx, i + 1) - flux(c, i, j, k, ex, u[i, j, k], nx, i)
    net += flux(c, i, j + 1, k, ey, v[i, j + 1, k], ny, j + 1) - flux(c, i, j, k, ey, v[i, j, k], ny, j)
    net += flux(c, i, j, k + 1, ez, w[i, j, k + 1], nz, k + 1) - flux(c, i, j, k, ez, w[i, j, k], nz, k)
    c0 = c[i, j, k]
    lap = diff(c, solid, i - 1, j, k, c0) + diff(c, solid, i + 1, j, k, c0) + diff(c, solid, i, j - 1, k, c0)
    lap += diff(c, solid, i, j + 1, k, c0) + diff(c, solid, i, j, k - 1, c0) + diff(c, solid, i, j, k + 1, c0)
    out[i, j, k] = c0 - h * net + kd * lap


@wp.kernel
def add_source(c: wp.array3d(dtype=float), cells: wp.array(dtype=wp.vec3i), weights: wp.array(dtype=float),
               amount: float):
    t = wp.tid()
    p = cells[t]
    wp.atomic_add(c, p[0], p[1], p[2], amount * weights[t])


@wp.kernel
def reduce_stats(c: wp.array3d(dtype=float), lo: float, hi: float, out: wp.array(dtype=float)):
    i, j, k = wp.tid()
    x = c[i, j, k]
    wp.atomic_add(out, 0, x)
    if x > lo:
        wp.atomic_add(out, 1, 1.0)
    if x > hi:
        wp.atomic_add(out, 2, 1.0)
    wp.atomic_max(out, 3, x)


@wp.kernel
def reduce_speed(f: wp.array3d(dtype=float), out: wp.array(dtype=float)):
    i, j, k = wp.tid()
    wp.atomic_max(out, 4, wp.abs(f[i, j, k]))
