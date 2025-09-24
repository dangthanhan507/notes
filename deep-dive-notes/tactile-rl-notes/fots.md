---
layout: default
title: FOTS
parent: Tactile RL Notes
mathjax: true
tags: 
  - latex
  - math
has_children: true
---


# FOTS

$$
\begin{equation}
M_c = M_{\text{ini}} + \Delta d_d + \Delta d_s + \Delta d_t
\end{equation}
$$

- $$M_c$$: current marker position
- $$M_{\text{ini}}$$: initial marker position (without object contact)
- $$\Delta d_d$$: dilate motion displacement
- $$\Delta d_s$$: shear motion displacement
- $$\Delta d_t$$: twist motion displacement

$$
\begin{align}
\Delta d_d &= \sum_{i=1}^N \Delta h_i \cdot (M - C_i) \cdot \exp(-\lambda_d \|M - C_i\|^2)
\end{align}
$$

- $$M$$: another way of saying $$M_{\text{ini}}$$
- $$\Delta h_i$$: height of $$C_i$$ (from depth map).
- $$C_i$$: markers in contact.
- $$N$$: number of $$C_i$$.

$$
\begin{align}
\Delta d_s &= \min\{\Delta s, \Delta s_{\max} \} \cdot \exp(-\lambda_s \| M - G \|^2_2) \\
\Delta d_t &= \min\{\Delta \theta, \Delta \theta_{\max} \} \cdot \exp(-\lambda_t \| M - G \|^2_2)
\end{align}
$$

- $$G$$: projection point of object coordinate system origin on gel surface along normal direction.
- $$\Delta s$$: translation distance of G relative to gel surface.
- $$\Delta \theta$$: rotation angle of object coordinate system relative to gel surface.

We calibrate $$\lambda_d, \lambda_s, \lambda_t$$ for FOTS.

## FOTS Code Implementation

This is the implementation of marker motion in FOTS.

```python
def __init__(....):
  ...

  # self.W, self.H are dimensions of image
  self.x = np.arange(0, self.W, 1)
  self.y = np.arange(0, self.H, 1)
  self.xx, self.yy = np.meshgrid(self.y, self.x)

def _marker_motion(self):
  xind = (np.random.random(self.N * self.M) * self.W).astype(np.int16)
  yind = (np.random.random(self.N * self.M) * self.H).astype(np.int32)

  x = np.arange(23, 320, 29)[:self.N]
  y = np.arange(15, 240, 26)[:self.M]
  
  xind, yind = np.meshgrid(x, y)
  xind = (xind.reshape([1, -1])[0]).astype(np.int16)
  yind = (yind.reshape([1, -1])[0]).astype(np.int16)

  xx_marker, yy_marker = self.xx[xind, yind].reshape([self.M, self.N]), self.yy[xind, yind].reshape([self.M, self.N])
  self.xx,self.yy = xx_marker, yy_marker
  img = self._generate(xx_marker, yy_marker)
  xx_marker_, yy_marker_ = self._motion_callback(xx_marker, yy_marker)
  img = self._generate(xx_marker_, yy_marker_)
  self.contact = []

def _motion_callback(self,xx,yy):
  for i in range(self.N):
      for j in range(self.M):
          r = int(yy[j, i])
          c = int(xx[j, i])
          if self.mask[r,c] == 1.0:
              h = self.depth[r,c]*100 # meter to mm
              self.contact.append([r,c,h])
  
  if not self.contact:
      xx,yy = self.xx,self.yy

  xx_,yy_ = self._dilate(self.lamb[0], xx ,yy)
  if len(self.traj) >= 2:
      xx_,yy_ = self._shear(int(self.traj[0][0]*meter2pix + 120), 
                          int(self.traj[0][1]*meter2pix + 160),
                          self.lamb[1],
                          int((self.traj[-1][0]-self.traj[0][0])*meter2pix),
                          int((self.traj[-1][1]-self.traj[0][1])*meter2pix),
                          xx_,
                          yy_)

      theta = max(min(self.traj[-1][2]-self.traj[0][2], 50 / 180.0 * math.pi), -50 / 180.0 * math.pi)
      xx_,yy_ = self._twist(int(self.traj[-1][0]*meter2pix + 120), 
                          int(self.traj[-1][1]*meter2pix + 160),
                          self.lamb[2],
                          theta,
                          xx_,
                          yy_)

  return xx_,yy_
```

# FOTS Calibration

For calibration (based on README.md of codebase), we take ~45 tactile flow images of a sphere indenter on different locations (in order of dilate, shear, and twist, 15 images for each type).

We get tactile flow through optical flow algorithm (Farneback) between initial image and deformed image.

We label center, circumference, and contact points (depth > 0) of dilate image. 


$$
\begin{align}
&\min_{\lambda_d} \sum_{d=1}^{|D|} \sum_{m=1}^{|M|} (y^{(d,m)} - f_d^{(d,m)}(\lambda_d))^2 \\
f_d(\lambda_d) &= \sum_{c=1}^C \Delta h_c \cdot (M - C_c) \cdot \exp(-\lambda_d \|M - C_c\|^2) \\
&\min_{\lambda_s} \sum_{s=1}^{|S|} \sum_{m=1}^{|M|} (y^{(s,m)} - f_s^{(s,m)}(\lambda_s))^2 \\
f_s(\lambda_s) &= \Delta s \cdot \exp(-\lambda_s \| M - G \|^2_2) \\
&\min_{\lambda_t} \sum_{t=1}^{|T|} \sum_{m=1}^{|M|} (y^{(t,m)} - f_t^{(t,m)}(\lambda_t))^2 \\
f_t(\lambda_t) &= \Delta \theta \cdot \exp(-\lambda_t \| M - G \|^2_2)
\end{align}
$$