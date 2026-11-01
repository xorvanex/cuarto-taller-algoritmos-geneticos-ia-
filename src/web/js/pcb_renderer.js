class PCBRenderer {
  constructor(canvas, instance) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.instance = instance;
    this.components = instance.components || [];
    this.nets = instance.nets || [];
    this.placed = [];
    this.hoveredComp = null;
    this.cssWidth = 400;
    this.cssHeight = 320;

    this._bindEvents();
  }

  _bindEvents() {
    this.canvas.addEventListener('mousemove', e => this._onMouseMove(e));
    this.canvas.addEventListener('mouseleave', () => {
      this.hoveredComp = null;
      const tooltip = document.getElementById('pcb-canvas-tooltip');
      if (tooltip) tooltip.style.display = 'none';
      this.draw(this.placed);
    });
  }

  _onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let found = null;
    if (this.placed && this.placed.length > 0) {
      for (const comp of this.placed) {
        const cp = this._toCanvas(comp.x, comp.y);
        const wPx = (comp.width / this.instance.board_max_width) * this.boardW;
        const hPx = (comp.height / this.instance.board_max_height) * this.boardH;
        if (
          mx >= cp.px - wPx / 2 &&
          mx <= cp.px + wPx / 2 &&
          my >= cp.py - hPx / 2 &&
          my <= cp.py + hPx / 2
        ) {
          found = comp;
          break;
        }
      }
    }

    if (found !== this.hoveredComp) {
      this.hoveredComp = found;
      this.draw(this.placed);
    }

    const tooltip = document.getElementById('pcb-canvas-tooltip');
    if (tooltip) {
      if (found) {
        // Find connected nets
        const connNets = this.nets.filter(n => n.source_id === found.id || n.target_id === found.id);
        const netList = connNets.map(n => n.net_name).slice(0, 5).join(', ');
        tooltip.innerHTML = `
          <div style="font-weight:700; color:${found.color || '#10B981'}; margin-bottom:4px">
            ${found.ref} — ${found.name}
          </div>
          <div style="color:#94A3B8; font-size:0.75rem; line-height:1.4">
            <strong>Huella:</strong> ${found.width} × ${found.height} mm (${found.category})<br>
            <strong>Posición:</strong> X = ${found.x.toFixed(1)} mm, Y = ${found.y.toFixed(1)} mm<br>
            <strong>Conexiones (${connNets.length}):</strong> <span style="color:#38BDF8">${netList || 'Ninguna'}</span>
          </div>
        `;
        tooltip.style.display = 'block';
        tooltip.style.left = `${Math.min(mx + 15, rect.width - 240)}px`;
        tooltip.style.top  = `${Math.max(my - 70, 10)}px`;
      } else {
        tooltip.style.display = 'none';
      }
    }
  }

  _toCanvas(x, y) {
    const pad = 30;
    const w = this.cssWidth;
    const h = this.cssHeight;

    const boardAspect = this.instance.board_max_width / this.instance.board_max_height; // 100 / 80 = 1.25
    let bW = w - 2 * pad;
    let bH = bW / boardAspect;
    if (bH > h - 2 * pad) {
      bH = h - 2 * pad;
      bW = bH * boardAspect;
    }

    this.boardW = bW;
    this.boardH = bH;
    this.boardX = (w - bW) / 2;
    this.boardY = (h - bH) / 2;

    return {
      px: this.boardX + (x / this.instance.board_max_width) * bW,
      py: this.boardY + (1 - y / this.instance.board_max_height) * bH, // Inverted Y for Cartesian board
    };
  }

  draw(placedComponents = null) {
    if (placedComponents) {
      this.placed = placedComponents;
    }

    const canvas = this.canvas;
    const ctx = this.ctx;
    const parent = canvas.parentElement;
    const rect = parent ? parent.getBoundingClientRect() : canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width) || 480;
    const h = Math.floor(rect.height) || 300;

    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    this.cssWidth = w;
    this.cssHeight = h;

    // Fondo general oscuro
    ctx.fillStyle = '#070D14';
    ctx.fillRect(0, 0, w, h);

    // Calcular dimensiones de la placa
    const origin = this._toCanvas(0, 0);
    const maxCorner = this._toCanvas(this.instance.board_max_width, this.instance.board_max_height);
    const bx = this.boardX;
    const by = this.boardY;
    const bw = this.boardW;
    const bh = this.boardH;

    // 1. Sustrato de la Placa PCB (FR-4 Solder Mask Verde Oscuro / Tech Teal)
    ctx.save();
    const bgGrad = ctx.createLinearGradient(bx, by, bx + bw, by + bh);
    bgGrad.addColorStop(0, '#0a1d17');
    bgGrad.addColorStop(1, '#06130f');
    ctx.fillStyle = bgGrad;
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = 2.0;

    // Sombra exterior de la placa
    ctx.shadowColor = 'rgba(16, 185, 129, 0.25)';
    ctx.shadowBlur = 12;

    this._roundRect(ctx, bx, by, bw, bh, 8);
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // 2. Grilla milimétrica sutil (cada 10 mm)
    ctx.save();
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.08)';
    ctx.lineWidth = 1;
    for (let gx = 10; gx < this.instance.board_max_width; gx += 10) {
      const p1 = this._toCanvas(gx, 0);
      const p2 = this._toCanvas(gx, this.instance.board_max_height);
      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();
    }
    for (let gy = 10; gy < this.instance.board_max_height; gy += 10) {
      const p1 = this._toCanvas(0, gy);
      const p2 = this._toCanvas(this.instance.board_max_width, gy);
      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();
    }
    ctx.restore();

    // 3. Orificios de Montaje en las 4 Esquinas (Mounting Holes)
    const mountHoles = [
      { x: 4, y: 4 },
      { x: this.instance.board_max_width - 4, y: 4 },
      { x: 4, y: this.instance.board_max_height - 4 },
      { x: this.instance.board_max_width - 4, y: this.instance.board_max_height - 4 }
    ];
    for (const h of mountHoles) {
      const cp = this._toCanvas(h.x, h.y);
      ctx.save();
      // Anillo de cobre dorado
      ctx.fillStyle = '#D97706';
      ctx.beginPath();
      ctx.arc(cp.px, cp.py, 5, 0, Math.PI * 2);
      ctx.fill();
      // Taladro interno oscuro
      ctx.fillStyle = '#070D14';
      ctx.beginPath();
      ctx.arc(cp.px, cp.py, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Si no hay componentes colocados, mostrar mensaje
    if (!this.placed || this.placed.length === 0) {
      ctx.fillStyle = '#64748B';
      ctx.font = '12px "Fira Code", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Cargando plano de colocación PCB...', w / 2, h / 2);
      return;
    }

    // Mapa de componentes por ID para rápido acceso
    const compMap = {};
    for (const c of this.placed) compMap[c.id] = c;

    // 4. Dibujar Trazas de Cobre (Nets)
    for (const net of this.nets) {
      const c1 = compMap[net.source_id];
      const c2 = compMap[net.target_id];
      if (!c1 || c2 === undefined) continue;

      const p1 = this._toCanvas(c1.x, c1.y);
      const p2 = this._toCanvas(c2.x, c2.y);

      const isHovered = this.hoveredComp && (this.hoveredComp.id === c1.id || this.hoveredComp.id === c2.id);

      ctx.save();
      if (isHovered) {
        ctx.strokeStyle = '#38BDF8';
        ctx.shadowColor = '#38BDF8';
        ctx.shadowBlur = 8;
        ctx.lineWidth = 2.5;
      } else {
        // Color por importancia del Net
        if (net.net_name.includes('VCC') || net.net_name.includes('VBUS') || net.net_name.includes('LDO')) {
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.75)'; // Alimentación ámbar
          ctx.lineWidth = 1.8;
        } else if (net.net_name.includes('XTAL')) {
          ctx.strokeStyle = 'rgba(139, 92, 246, 0.75)'; // Reloj púrpura
          ctx.lineWidth = 1.6;
        } else if (net.net_name.includes('SPI') || net.net_name.includes('USB')) {
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.70)'; // Datos cyan
          ctx.lineWidth = 1.4;
        } else {
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.50)'; // General esmeralda
          ctx.lineWidth = 1.2;
        }
      }

      // Ruteo ortogonal Manhattan con chaflán
      const midX = p1.px + (p2.px - p1.px) * 0.5;
      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(midX, p1.py);
      ctx.lineTo(midX, p2.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();

      // Pequeñas vías en esquinas
      ctx.fillStyle = isHovered ? '#38BDF8' : '#D97706';
      ctx.beginPath();
      ctx.arc(midX, p1.py, 1.8, 0, Math.PI * 2);
      ctx.arc(midX, p2.py, 1.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // 5. Bounding Box Envolvente (Área compacta del circuito)
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const c of this.placed) {
      const halfW = c.width / 2;
      const halfH = c.height / 2;
      minX = Math.min(minX, c.x - halfW);
      maxX = Math.max(maxX, c.x + halfW);
      minY = Math.min(minY, c.y - halfH);
      maxY = Math.max(maxY, c.y + halfH);
    }
    const bbP1 = this._toCanvas(minX, maxY);
    const bbP2 = this._toCanvas(maxX, minY);

    ctx.save();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.40)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(bbP1.px - 3, bbP1.py - 3, (bbP2.px - bbP1.px) + 6, (bbP2.py - bbP1.py) + 6);
    ctx.restore();

    // 6. Dibujar Componentes Electrónicos (Footprints & Silkscreen)
    for (const comp of this.placed) {
      const cp = this._toCanvas(comp.x, comp.y);
      const wPx = (comp.width / this.instance.board_max_width) * this.boardW;
      const hPx = (comp.height / this.instance.board_max_height) * this.boardH;
      const isHovered = this.hoveredComp && this.hoveredComp.id === comp.id;

      ctx.save();

      // Halo de aislamiento dieléctrico (clearance 2mm)
      const clrWPx = (this.instance.clearance / this.instance.board_max_width) * this.boardW;
      const clrHPx = (this.instance.clearance / this.instance.board_max_height) * this.boardH;
      ctx.strokeStyle = isHovered ? 'rgba(56, 189, 248, 0.4)' : 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.strokeRect(cp.px - wPx / 2 - clrWPx, cp.py - hPx / 2 - clrHPx, wPx + 2 * clrWPx, hPx + 2 * clrHPx);
      ctx.setLineDash([]);

      // Cuerpo del componente según tipo
      ctx.shadowColor = isHovered ? (comp.color || '#10B981') : 'rgba(0, 0, 0, 0.5)';
      ctx.shadowBlur = isHovered ? 12 : 4;

      if (comp.category === 'ic') {
        // Chip IC (MCU, Flash)
        ctx.fillStyle = '#1E293B';
        ctx.strokeStyle = comp.color || '#10B981';
        ctx.lineWidth = isHovered ? 2 : 1.2;
        this._roundRect(ctx, cp.px - wPx / 2, cp.py - hPx / 2, wPx, hPx, 3);
        ctx.fill();
        ctx.stroke();

        // Pin 1 dot
        ctx.fillStyle = '#F8FAFC';
        ctx.beginPath();
        ctx.arc(cp.px - wPx / 2 + 3, cp.py - hPx / 2 + 3, 1.5, 0, Math.PI * 2);
        ctx.fill();

        // Pins a los costados
        ctx.fillStyle = '#94A3B8';
        const numPins = 4;
        for (let i = 0; i < numPins; i++) {
          const step = hPx / (numPins + 1);
          const py = cp.py - hPx / 2 + (i + 1) * step;
          ctx.fillRect(cp.px - wPx / 2 - 2, py - 1, 2, 2);
          ctx.fillRect(cp.px + wPx / 2, py - 1, 2, 2);
        }
      } else if (comp.category === 'power') {
        // Regulador LDO / Diodo
        ctx.fillStyle = '#1e2433';
        ctx.strokeStyle = comp.color || '#F59E0B';
        ctx.lineWidth = isHovered ? 2 : 1.2;
        this._roundRect(ctx, cp.px - wPx / 2, cp.py - hPx / 2, wPx, hPx, 3);
        ctx.fill();
        ctx.stroke();

        // Heatsink tab
        ctx.fillStyle = '#D97706';
        ctx.fillRect(cp.px - wPx / 3, cp.py - hPx / 2 - 2, (wPx * 2) / 3, 2);
      } else if (comp.category === 'connector') {
        // Conector USB / GPIO
        ctx.fillStyle = '#172554';
        ctx.strokeStyle = comp.color || '#3B82F6';
        ctx.lineWidth = isHovered ? 2 : 1.2;
        this._roundRect(ctx, cp.px - wPx / 2, cp.py - hPx / 2, wPx, hPx, 2);
        ctx.fill();
        ctx.stroke();

        // Pines de cabezal o conector
        ctx.fillStyle = '#E2E8F0';
        for (let ix = -wPx / 3; ix <= wPx / 3; ix += wPx / 3) {
          ctx.beginPath();
          ctx.arc(cp.px + ix, cp.py, 1.5, 0, Math.PI * 2);
          ctx.fill();
        }
      } else {
        // Componentes pasivos (Capacitores, Resistencias 0805)
        ctx.fillStyle = '#0f172a';
        ctx.strokeStyle = comp.color || '#EC4899';
        ctx.lineWidth = isHovered ? 1.8 : 1.0;
        this._roundRect(ctx, cp.px - wPx / 2, cp.py - hPx / 2, wPx, hPx, 1.5);
        ctx.fill();
        ctx.stroke();

        // Terminales plateadas
        ctx.fillStyle = '#CBD5E1';
        ctx.fillRect(cp.px - wPx / 2, cp.py - hPx / 2, 2, hPx);
        ctx.fillRect(cp.px + wPx / 2 - 2, cp.py - hPx / 2, 2, hPx);
      }

      // Serigrafía: Ref del componente (Silkscreen blanca / menta)
      ctx.fillStyle = isHovered ? '#FFFFFF' : '#E2E8F0';
      const fontSize = Math.max(8, Math.min(11, Math.round(wPx * 0.35)));
      ctx.font = `700 ${fontSize}px "Fira Code", monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(comp.ref, cp.px, cp.py);

      ctx.restore();
    }
  }

  _roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }
}


/* ════════════════════════════════════════════════════════════
   12. PROBLEMA 3: MARKOWITZ — Frontera Eficiente y Distribución
   ════════════════════════════════════════════════════════════ */

const MARKOWITZ_PY_DATA = typeof window !== 'undefined' && window.MARKOWITZ_SIMULATION_DATA ? window.MARKOWITZ_SIMULATION_DATA : null;

