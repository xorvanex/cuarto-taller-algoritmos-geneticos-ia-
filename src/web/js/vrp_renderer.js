class VRPRenderer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {typeof VRP_INSTANCE} instance
   */
  constructor(canvas, instance) {
    this.canvas   = canvas;
    this.ctx      = canvas.getContext('2d');
    this.instance = instance;
    this.hoveredNode = null;   // índice del cliente bajo el cursor (-1 = depot)

    this._bindEvents();
    this.draw(null, []);
  }

  /**
   * Convierte coordenadas del problema [0-100] → píxeles CSS del canvas,
   * manteniendo un plano cartesiano perfectamente cuadrado (1:1) y centrado.
   */
  _toCanvas(x, y) {
    const pad = 28;
    const w = this.cssWidth || 400;
    const h = this.cssHeight || 300;
    const side = Math.min(w - 2 * pad, h - 2 * pad);
    const offsetX = (w - side) / 2;
    const offsetY = (h - side) / 2;

    return {
      px: offsetX + (x / 100) * side,
      py: offsetY + (1 - y / 100) * side,  // Y invertido para plano cartesiano
    };
  }

  _bindEvents() {
    this.canvas.addEventListener('mousemove', e => this._onMouseMove(e));
    this.canvas.addEventListener('mouseleave', () => {
      this.hoveredNode = null;
      const tooltip = document.getElementById('canvas-tooltip');
      if (tooltip) tooltip.style.display = 'none';
    });
  }

  _onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const my   = e.clientY - rect.top;

    let closest = null;
    let minD    = 18;  // radio de detección en px CSS

    // Depot
    const dep = this._toCanvas(this.instance.depot.x, this.instance.depot.y);
    const dDep = Math.hypot(mx - dep.px, my - dep.py);
    if (dDep < minD) { closest = { type: 'depot' }; minD = dDep; }

    // Clientes
    this.instance.customers.forEach((c, i) => {
      const p = this._toCanvas(c.x, c.y);
      const d = Math.hypot(mx - p.px, my - p.py);
      if (d < minD) { closest = { type: 'customer', index: i }; minD = d; }
    });

    const tooltip = document.getElementById('canvas-tooltip');
    if (!tooltip) return;

    if (closest) {
      let label = '';
      if (closest.type === 'depot') {
        label = `Depósito Central (${this.instance.depot.x}, ${this.instance.depot.y})`;
      } else {
        const c = this.instance.customers[closest.index];
        label = `${c.id} | Dem: ${c.demand} kg | Serv: ${c.service} min`;
      }
      tooltip.style.display = 'block';
      tooltip.style.left = (e.offsetX + 14) + 'px';
      tooltip.style.top  = (e.offsetY - 28) + 'px';
      tooltip.textContent = label;
    } else {
      tooltip.style.display = 'none';
    }
    this.hoveredNode = closest;
  }

  /**
   * Dibuja el mapa de rutas VRP centrado y escalado.
   * @param {number[][]} routes   - rutas actuales (puede ser null)
   * @param {string[]}   [routeColors]
   */
  draw(routes, routeColors = ROUTE_COLORS) {
    const canvas = this.canvas;
    const ctx    = this.ctx;

    // Obtener dimensiones reales del contenedor padre (.panel-body)
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr  = window.devicePixelRatio || 1;
    const w    = Math.floor(rect.width) || 400;
    const h    = Math.floor(rect.height) || 300;

    // Ajustar resolución del canvas para retina displays
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width  = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    this.cssWidth  = w;
    this.cssHeight = h;

    ctx.clearRect(0, 0, w, h);

    // Fondo completo de la tarjeta
    ctx.fillStyle = 'rgba(7, 11, 18, 0.75)';
    ctx.fillRect(0, 0, w, h);

    // Cuadrícula 1:1 centrada
    this._drawGrid(ctx, w, h);

    // Rutas
    if (routes) {
      routes.forEach((route, v) => {
        if (!route || route.length === 0) return;
        this._drawRoute(ctx, route, routeColors[v] || '#fff');
      });
    }

    // Clientes
    this.instance.customers.forEach((c, i) => {
      this._drawCustomer(ctx, c, i, routes);
    });

    // Depósito
    this._drawDepot(ctx);
  }

  _drawGrid(ctx, w, h) {
    ctx.save();
    const pad = 28;
    const side = Math.min(w - 2 * pad, h - 2 * pad);
    const offsetX = (w - side) / 2;
    const offsetY = (h - side) / 2;

    // Recuadro interior centrado con brillo tenue
    ctx.fillStyle = 'rgba(11, 15, 25, 0.8)';
    ctx.fillRect(offsetX, offsetY, side, side);

    // Cuadrícula 10x10
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth   = 1;
    const step = side / 10;

    for (let i = 0; i <= 10; i++) {
      const pos = offsetX + i * step;
      ctx.beginPath();
      ctx.moveTo(pos, offsetY);
      ctx.lineTo(pos, offsetY + side);
      ctx.stroke();

      const yPos = offsetY + i * step;
      ctx.beginPath();
      ctx.moveTo(offsetX, yPos);
      ctx.lineTo(offsetX + side, yPos);
      ctx.stroke();
    }

    // Marco exterior de la zona logística
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.25)';
    ctx.lineWidth   = 1.5;
    ctx.strokeRect(offsetX, offsetY, side, side);

    ctx.restore();
  }

  _drawRoute(ctx, route, color) {
    const inst = this.instance;
    const points = [
      this._toCanvas(inst.depot.x, inst.depot.y),
      ...route.map(ci => this._toCanvas(inst.customers[ci].x, inst.customers[ci].y)),
      this._toCanvas(inst.depot.x, inst.depot.y),
    ];

    ctx.save();
    ctx.shadowColor  = color;
    ctx.shadowBlur   = 8;
    ctx.strokeStyle  = color + 'CC';
    ctx.lineWidth    = 2.2;
    ctx.lineCap      = 'round';
    ctx.lineJoin     = 'round';

    ctx.beginPath();
    points.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p.px, p.py);
      else ctx.lineTo(p.px, p.py);
    });
    ctx.stroke();

    // Flechas direccionales
    for (let i = 0; i < points.length - 1; i++) {
      this._drawArrow(ctx, points[i], points[i + 1], color);
    }
    ctx.restore();
  }

  _drawArrow(ctx, from, to, color) {
    const mx  = (from.px + to.px) / 2;
    const my  = (from.py + to.py) / 2;
    const ang = Math.atan2(to.py - from.py, to.px - from.px);
    const len = 7;

    ctx.save();
    ctx.translate(mx, my);
    ctx.rotate(ang);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(len, 0);
    ctx.lineTo(-len * 0.7, -len * 0.5);
    ctx.lineTo(-len * 0.7,  len * 0.5);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  _drawCustomer(ctx, c, i, routes) {
    const { px, py } = this._toCanvas(c.x, c.y);
    const isHovered  = this.hoveredNode?.type === 'customer' && this.hoveredNode.index === i;

    // Determinar qué camión atiende al cliente
    let color = '#64748B';
    if (routes) {
      routes.forEach((route, v) => {
        if (route.includes(i)) color = ROUTE_COLORS[v] || '#fff';
      });
    }

    ctx.save();
    if (isHovered) {
      ctx.shadowColor = color;
      ctx.shadowBlur  = 16;
      ctx.beginPath();
      ctx.arc(px, py, 13, 0, Math.PI * 2);
      ctx.fillStyle = color + '33';
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(px, py, isHovered ? 9 : 7, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 6;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#070B12';
    ctx.fill();

    // Etiqueta del nodo
    ctx.fillStyle    = '#F8FAFC';
    ctx.font         = 'bold 9px "Fira Code", monospace';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(c.id, px, py - 9);

    ctx.restore();
  }

  _drawDepot(ctx) {
    const { px, py } = this._toCanvas(this.instance.depot.x, this.instance.depot.y);
    const isHovered  = this.hoveredNode?.type === 'depot';
    const sz         = isHovered ? 12 : 9;

    ctx.save();
    if (isHovered) {
      ctx.shadowColor = '#10B981';
      ctx.shadowBlur  = 20;
    }

    ctx.translate(px, py);
    ctx.rotate(Math.PI / 4);

    ctx.fillStyle   = '#0B0F19';
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth   = 2.2;
    ctx.fillRect(-sz, -sz, sz * 2, sz * 2);
    ctx.strokeRect(-sz, -sz, sz * 2, sz * 2);

    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle    = '#10B981';
    ctx.font         = 'bold 8px "Inter", sans-serif';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('D', 0, 0);

    ctx.restore();
  }
}

/* ════════════════════════════════════════════════════════════
   8. CHART.JS — Gráfica de Convergencia
   ════════════════════════════════════════════════════════════ */

