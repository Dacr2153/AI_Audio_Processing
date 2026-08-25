# Plan de Mejora — TUI Design System Redesign

## Contexto del Proyecto

**Proyecto:** `audio_restoration` — Herramienta CLI/TUI para restauración de audio con pipeline de DSP y modelos neuronales.

**Stack:**
- Python 3.14, Textual (TUI framework), asyncio
- Tests: pytest + pytest-asyncio (headless Textual app)
- Linting: ruff, mypy
- Architecture: `TuiScreen(Vertical)` base class → `compose()` titulo + `form()` yield

**Archivos clave:**
```
src/audio_restoration/tui/
├── app.py                    # App principal, themes, CommandBar
├── design.py                 # Design tokens (spacing, button roles)
├── state.py                  # TuiState (config, profiles, history)
├── i18n.py                   # Tabla de strings bilingual EN/ES
├── navigation.py             # SCREENS dict, ScreenRequested message
├── components/
│   ├── sidebar.py            # Sidebar compacto con nav rail
│   ├── footer.py             # CommandBar contextual
│   ├── form.py               # FieldRow (label + input/browse)
│   ├── file_picker.py        # FilePickerScreen ModalScreen
│   └── results.py            # ResultsPanel para metricas
└── screens/
    ├── base.py               # TuiScreen base (compose + form pattern)
    ├── home.py               # Dashboard con quick actions
    ├── single.py             # Restauracion archivo individual
    ├── batch.py              # Restauracion por lotes
    ├── profiles.py           # CRUD de perfiles
    ├── history.py            # Tabla de historial
    └── about.py              # Version, deps, licencia
```

---

## Fases Completadas

### Fase 1: Design System — Paleta + Tokens ✅
**Commit:** `810d228`
- Nueva paleta de colores: `#E5A24F` (accent), `#0B0E12` (bg), `#15191E` (surface), `#1C2028` (surface-active)
- `design.py`: tokens `SPACING_XS..XL`, `BUTTON_PRIMARY/SECONDARY/TERTIARY`
- Themes dark/light definidos en `app.py`

### Fase 2: Navigation — Sidebar + CommandBar ✅
**Commit:** `8649cc3`
- Sidebar compacto con botones flat, clase `-active`, separadores, version
- `CommandBar` footer contextual (muestra shortcuts por pantalla)

### Fase 3: Componentes Base — Botones, Inputs, Paneles ✅
**Commit:** `a6cd647`
- Jerarquía de botones: `Button.primary`, `Button.default`, `Button.flat`
- Inputs compactos con focus border `$accent`
- Clase `.panel` para contenedores con borde sutil
- `FieldRow` con spacing mejorado

### Fase 4: Home Dashboard ✅
**Commit:** `c9d2e4a`
- Quick actions (botones a Single/Batch)
- Recent activity (ultimas 5 entradas del historial)
- Active profile (resumen de config actual)

### Fase 5: Single Screen — Paneles Agrupados ✅
**Commit:** `96ff033`
- Paneles `.file-panel` para input/output
- `FieldRow(label_key=None)` para labels manuales
- Action row con boton primary + hint

### Fase 6: Batch — Tabla Mejorada ✅
**Commit:** `db6fab4`
- Folder panels para input/output
- Options panel (ext, suffix, workers)
- DataTable con `zebra_stripes=True`
- Summary panel con resultados

### Fase 7: Profiles, History, About ✅
**Commit:** `111c494`
- Profiles: save-panel con input + boton
- History: action-row con boton danger + hint
- About: paneles separados (version, deps, license, repo)

---

## Fases Pendientes

### Fase 8: Estabilizar Tests Flaky 🔴 Prioridad Alta

**Problema:** Los tests de TUI fallan aleatoriamente (diferente test cada ejecucion). No es un problema de codigo sino de aislamiento entre tests.

**Causa raíz:** Los tests comparten estado global (config, profiles, history files) y Textual headless mode tiene timing issues con `on_mount()` y `worker_state_changed`.

**Archivos afectados:**
- `tests/test_tui/test_single.py`
- `tests/test_tui/test_batch.py`
- `tests/test_tui/test_profiles.py`
- `tests/test_tui/test_history.py`
- `tests/test_tui/test_about.py`

**Plan de accion:**

1. **Aislar fixtures con `tmp_path`:**
   - Cada test debe usar directorio temporal para config/profiles/history
   - Monkeypatch `Path.home()` o rutas de archivos para evitar colisiones
   - Fixture `isolated_state` que crea `TuiState` con paths temporales

2. **Fixtures de Textual App:**
   - Crear fixture `app_factory` que retorna instancia limpia por test
   - Usar `async with app.run_test()` context manager para cada test
   - Asegurar que `on_mount()` completa antes de interactuar

3. **Patron para tests de screens:**
   ```python
   @pytest.fixture
   def isolated_state(tmp_path, monkeypatch):
       """State aislado para tests."""
       monkeypatch.setattr(
           "audio_restoration.config_serde.CONFIG_DIR",
           tmp_path / "config"
       )
       return TuiState()

   async def test_something(isolated_state):
       app = TuiApp(state=isolated_state)
       async with app.run_test() as pilot:
           # interactuar con pilot
           await pilot.pause()
   ```

4. **Agregar `await pilot.pause()` despues de acciones:**
   - Despues de `pilot.click()`
   - Despues de `pilot.press("enter")`
   - Antes de assertions que dependen de renders

5. **Verificar:** Correr `pytest tests/test_tui/ -q` 10 veces seguidas sin fallas

---

### Fase 9: Scroll Verificacion y Polish 🔴 Prioridad Alta

**Problema original:** Los campos de output no se veian porque estaban fuera del viewport.

**Solucion implementada:** `overflow-y: auto` en `TuiScreen > Vertical` (base.py).

**Verificaciones pendientes:**

1. **Probar manualmente** con不同 screen sizes:
   - Terminal 80x24 (minimo)
   - Terminal 120x40 (comodo)
   - Verificar que scroll funciona en Single y Batch

2. **Verificar que `height: 1fr` no rompa nada:**
   - El `Vertical` interno debe occupar todo el espacio disponible
   - Los panels con `height: auto` deben expandirse correctamente
   - Los DataTables con `height: 1fr` deben ser scrollables

3. **Ajustar si es necesario:**
   - Si el scroll es muy agresivo, ajustar `margin-bottom` de panels
   - Si los panels no se ven compactos, reducir padding
   - Verificar que el CommandBar no se superponga al contenido

---

### Fase 10: Keyboard Navigation Completa 🟡 Prioridad Media

**Estado actual:** Solo hay bindings basicos (`ctrl+r` run, `ctrl+s` save, etc.)

**Mejoras necesarias:**

1. **Tab navigation:**
   - Tab debe mover foco entre Input fields
   - Shift+Tab debe mover foco hacia atras
   - Enter en un Input debe mover al siguiente Input (no submitir form)

2. **Quick actions en Home:**
   - `1` o `s` para Single, `2` o `b` para Batch
   - `h` para History, `p` para Profiles, `a` para About

3. **Batch table:**
   - Flechas arriba/abajo para navegar filas
   - Enter en una fila podria mostrar detalle (futuro)

4. **Profiles table:**
   - Enter para cargar perfil seleccionado
   - Delete para eliminar perfil seleccionado

**Archivos a modificar:**
- `src/audio_restoration/tui/screens/base.py` (bindings globales)
- `src/audio_restoration/tui/screens/home.py`
- `src/audio_restoration/tui/screens/single.py`
- `src/audio_restoration/tui/screens/batch.py`
- `src/audio_restoration/tui/screens/profiles.py`
- `src/audio_restoration/tui/components/sidebar.py`

---

### Fase 11: Loading States y Feedback Visual 🟡 Prioridad Media

**Problema:** Cuando el pipeline corre, el usuario no ve progreso claro.

**Mejoras:**

1. **Spinner/ProgressBar en Single:**
   - Mientras corre el pipeline, mostrar spinner animado
   - Cambiar boton "Run" a "Processing..." con spinner

2. **Progress en Batch:**
   - Barra de progreso global (X de Y archivos)
   - Indicador de archivo actual
   - ETA estimado

3. **Loading en About:**
   - Spinner mientras se verifican dependencias
   - Checkmarks verdes / Rojos para cada dependencia

4. **Notificaciones:**
   - Usar `self.notify()` con colores apropiados
   - Success: verde, Warning: amarillo, Error: rojo

**Archivos a modificar:**
- `src/audio_restoration/tui/screens/single.py`
- `src/audio_restoration/tui/screens/batch.py`
- `src/audio_restoration/tui/screens/about.py`
- Posiblemente crear `src/audio_restoration/tui/components/spinner.py`

---

### Fase 12: Responsive Design 🟢 Prioridad Baja

**Problema:** En terminales pequeñas (< 80 cols), el layout se rompe.

**Mejoras:**

1. **Sidebar colapsable:**
   - En terminales < 80 cols, sidebar se colapsa a solo iconos
   - En terminales < 60 cols, sidebar desaparece (mostrar con Ctrl+B)

2. **Panels responsive:**
   - FieldRows deben usar `width: 1fr` en vez de anchos fijos
   - Botones deben ser `compact=True` siempre
   - Tables deben tener `max-width: 100%`

3. **Breakpoints:**
   ```css
   TuiScreen { /* default: desktop */ }
   @media (max-width: 80) { /* tablet */ }
   @media (max-width: 60) { /* mobile */ }
   ```

**Nota:** Textual tiene soporte limitado para media queries. Esto podria requerir cambios en el Python code (detectar terminal size en `on_mount` y ajustar CSS dinamicamente).

---

### Fase 13: Theming Dinámico 🟢 Prioridad Baja

**Estado actual:** Dark y Light themes estaticos.

**Mejoras:**

1. **Toggle con Ctrl+D:**
   - Alternar entre dark/light sin reiniciar app
   - Guardar preferencia en config

2. **Custom themes:**
   - Permitir al usuario definir accent color
   - Exportar/importar themes como JSON

3. **High contrast mode:**
   - Para accesibilidad
   - Colores con alto contraste (WCAG AA)

---

### Fase 14: Documentación y Polish Final 🟢 Prioridad Baja

1. **Actualizar README:**
   - Screenshots del TUI funcionando
   - Tabla de shortcuts completa
   - Guia de uso por pantalla

2. **Docstrings:**
   - Asegurar que todos los metodos publicos tengan docstrings
   - Explicar patron `form()` en base class

3. **Type hints:**
   - Verificar que todos los retornos estan tipados
   - Eliminar `# type: ignore` innecesarios

4. **Changelog:**
   - Documentar todas las mejoras del design system
   - Incluir breaking changes (si los hay)

---

## Orden de Ejecución Recomendado

```
Fase 8 (Tests) → Fase 9 (Scroll) → Fase 10 (Keyboard) → Fase 11 (Loading)
     ↓                                                    ↓
Fase 12 (Responsive) ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← Fase 13 (Theming)
     ↓
Fase 14 (Docs)
```

**Razon:** 
- Fase 8 primero porque sin tests estables no se puede verificar nada
- Fase 9 porque es el bug reportado por el usuario
- Fases 10-11 mejoran UX significativamente
- Fases 12-14 son polish que puede esperar

---

## Comandos Útiles

```bash
# Ejecutar todos los tests
venv_audio/bin/pytest -q

# Ejecutar solo tests de TUI
venv_audio/bin/pytest tests/test_tui/ -q

# Ejecutar tests flaky multiples veces
venv_audio/bin/pytest tests/test_tui/ -q --count=10

# Verificar lint
venv_audio/bin/ruff check src/audio_restoration/tui/

# Verificar tipos
venv_audio/bin/mypy src/audio_restoration/tui/

# Ejecutar TUI
venv_audio/bin/python -m audio_restoration tui
```

---

## Notas para IA

1. **Patron de screens:** Todos los screens heredan de `TuiScreen` e implementan `form()` que yield widgets. El titulo se renderea automaticamente.

2. **i18n:** Todos los strings visibles usan `i18n.t("key")`. Las keys estan en `i18n.py`. Al agregar strings nuevos, agregar keys en ambos idiomas.

3. **Tests:** Los tests usan `app.run_test()` context manager de Textual. Siempre usar `await pilot.pause()` despues de acciones async.

4. **CSS:** Textual CSS es similar a web CSS pero con restricciones. No usar `px` ni `%`. Usar unidades de carattere (números enteros o `fr`).

5. **State:** `TuiState` es el centro de datos. Acceder via `self.state` en screens. Mutation methods: `save_profile()`, `add_history_entry()`, etc.

6. **Bindings:** Definir en `BINDINGS: ClassVar[list]` en la clase del screen. El `CommandBar` los muestra automaticamente.
