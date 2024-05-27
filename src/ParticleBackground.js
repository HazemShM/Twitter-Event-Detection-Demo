import React, { useCallback } from 'react';
import Particles from 'react-tsparticles';
import { loadFull } from 'tsparticles';

const ParticleBackground = () => {
  const particlesInit = useCallback(async (engine) => {
    await loadFull(engine);
  }, []);

  const particleOptions = {
    background: {
      color: {
        value: "#f4f4f9",
      },
    },
    fpsLimit: 60,
    interactivity: {
      events: {
        onClick: {
          enable: true,
          mode: "push",
        },
        onHover: {
          enable: true,
          mode: "repulse",
        },
        resize: true,
      },
      modes: {
        push: {
          quantity: 4,
        },
        repulse: {
          distance: 200,
          duration: 0.4,
        },
      },
    },
    particles: {
      color: {
        value: "#1DA1F2",
      },
      links: {
        color: "#1DA1F2",
        distance: 150,
        enable: true,
        opacity: 0.5,
        width: 1,
      },
      collisions: {
        enable: true,
      },
      move: {
        direction: "none",
        enable: true,
        outModes: {
          default: "bounce",
        },
        random: false,
        speed: 1.5,
        straight: false,
      },
      number: {
        density: {
          enable: true,
          area: 800,
        },
        value: 80,
      },
      opacity: {
        value: 0.5,
      },
      shape: {
        type: "circle",
      },
      size: {
        value: { min: 1, max: 5 },
      },
    },
    absorbers: [
      {
        position: { x: 50, y: 20 },
        size: { radius: 50, random: false },
        opacity: 0, 
      },
      {
        position: { x: 50, y: 85 },
        size: { radius: 50, random: false },
        opacity: 0, 
      },
    ],
    detectRetina: true,
  };

  return <Particles id="tsparticles" init={particlesInit} options={particleOptions} />;
};

export default ParticleBackground;
