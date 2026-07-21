"""Высокоуровневый решатель одномерной задачи на собственные значения.

В этом модуле низкоуровневая рекурсия Нумерова превращается в полноценный
алгоритм решения уравнения Шрёдингера:

1. для пробной энергии строятся левая и правая ветви волновой функции;
2. в точке сшивки вычисляется их вронскиан;
3. собственные энергии находятся как нули mismatch-функции;
4. две ветви масштабируются, объединяются и нормируются.

Для надёжного выделения близких уровней дополнительно используется дешёвая
конечно-разностная оценка спектра через последовательность Штурма. Она служит
только для выбора интервалов поиска; финальные энергии всегда уточняются по
нулю Numerov mismatch-функции.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .numerov import (
    UniformGrid,
    derivative_from_left,
    derivative_from_right,
    integrate_from_left,
    integrate_from_right,
)


# Типы используются только для статической проверки и подсказок IDE.
FloatArray = NDArray[np.float64]
Potential = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class Eigenstate:
    """Контейнер для одного нормированного стационарного состояния.

    Attributes
    ----------
    index:
        Номер уровня `n`, начиная с нуля.
    energy:
        Найденная собственная энергия `E_n`.
    x:
        Пространственная сетка.
    psi:
        Нормированная волновая функция на этой сетке.
    residual:
        Абсолютное значение нормированной mismatch-функции при найденной
        энергии. Чем оно меньше, тем точнее сошлись левая и правая ветви.
    """

    index: int
    energy: float
    x: FloatArray
    psi: FloatArray
    residual: float

    @property
    def probability_density(self) -> FloatArray:
        """Вернуть плотность вероятности `|psi(x)|^2`.

        В рассматриваемой задаче волновые функции вещественны, поэтому
        `|psi|^2 = psi**2`.
        """

        return self.psi**2


class MatchedNumerovSolver:
    """Решить `-psi''/2 + V(x)psi = Epsi` в единицах `ℏ=m=1`.

    Parameters
    ----------
    potential:
        Любая вызываемая Python-функция, принимающая массив координат и
        возвращающая значения `V(x)`.
    grid:
        Равномерная конечная сетка. Её границы должны находиться глубоко в
        классически запрещённой области для всех искомых энергий.
    match_x:
        Необязательная координата сшивки. Если она не задана, код пытается
        найти центральный максимум между двумя наиболее глубокими минимумами.
    seed:
        Произвольное малое ненулевое значение рядом с каждой границей. Оно
        запускает двухшаговую рекурсию Нумерова и не задаёт нормировку.
    """

    def __init__(
        self,
        potential: Potential,
        grid: UniformGrid,
        *,
        match_x: float | None = None,
        seed: float = 1.0e-12,
    ) -> None:
        """Подготовить сетку, потенциал и индекс точки сшивки.

        Здесь не ищутся собственные значения. Конструктор только вычисляет все
        величины, не зависящие от энергии, и проверяет входные данные.
        """

        self.potential = potential
        self.grid = grid
        self.x = grid.x
        self.h = grid.step
        self.seed = float(seed)

        # Потенциал вычисляется сразу на всей пространственной сетке. Это
        # быстрее, чем вызывать пользовательскую функцию внутри каждого шага
        # интегрирования для каждой пробной энергии.
        values = np.asarray(potential(self.x), dtype=float)

        # Некоторые функции возвращают скаляр, например постоянный потенциал.
        # В таком случае NumPy может растянуть его на всю сетку. Если форма
        # несовместима, выдаём понятную ошибку.
        if values.shape != self.x.shape:
            try:
                values = np.broadcast_to(values, self.x.shape).astype(float)
            except ValueError as exc:
                raise ValueError(
                    "Potential callable returned an incompatible shape"
                ) from exc

        if not np.all(np.isfinite(values)):
            raise ValueError("Potential must be finite on the whole grid")
        self.v = values

        # Пользовательская координата переводится в индекс ближайшего узла.
        # Если координата не задана, выбирается подходящая точка автоматически.
        if match_x is None:
            self.match_index = self._suggest_match_index()
        else:
            self.match_index = int(np.argmin(np.abs(self.x - match_x)))

        # Для односторонних производных четвёртого порядка по обе стороны от
        # точки сшивки требуется не менее четырёх соседних узлов.
        if not 4 <= self.match_index <= len(self.x) - 5:
            raise ValueError(
                "Matching point must be at least four grid cells from each edge"
            )

    @property
    def match_x(self) -> float:
        """Вернуть фактическую координату узла, используемого для сшивки."""

        return float(self.x[self.match_index])

    def _suggest_match_index(self) -> int:
        """Автоматически выбрать индекс точки сшивки.

        Для двойной ямы удобнее всего сшивать решения около центрального
        барьера: обе ветви проходят сравнительно короткий путь через внешние
        запрещённые области, а численная неустойчивость меньше.

        Алгоритм:
        1. находит все дискретные локальные минимумы потенциала;
        2. выбирает два минимума с наименьшими значениями `V`;
        3. между ними выбирает узел с максимальным потенциалом.

        Если структура потенциала не похожа на двойную яму, используется центр
        расчётной области как универсальный запасной вариант.
        """

        # Условие строгого локального минимума сравнивает каждый внутренний
        # узел с левым и правым соседом.
        minima = np.where(
            (self.v[1:-1] < self.v[:-2]) & (self.v[1:-1] < self.v[2:])
        )[0] + 1

        if len(minima) >= 2:
            # Из всех минимумов выбираются два самых глубоких.
            two_lowest = minima[np.argsort(self.v[minima])[:2]]
            left, right = np.sort(two_lowest)

            if right - left >= 2:
                # `argmax` возвращает индекс внутри среза, поэтому добавляем
                # левую границу среза, чтобы получить глобальный индекс.
                return int(left + np.argmax(self.v[left : right + 1]))

        # Универсальный запасной вариант для одноямных или необычных
        # потенциалов: ближайший узел к геометрическому центру области.
        centre = 0.5 * (self.grid.x_min + self.grid.x_max)
        return int(np.argmin(np.abs(self.x - centre)))

    def _branches(
        self, energy: float
    ) -> tuple[FloatArray, FloatArray, float, float]:
        """Построить левую и правую ветви для одной пробной энергии.

        Уравнение Шрёдингера переписывается в виде

            psi'' + k(x;E) psi = 0,
            k(x;E) = 2(E - V(x)).

        После интегрирования вычисляются односторонние производные обеих ветвей
        в точке сшивки. Эти четыре объекта полностью определяют mismatch.

        Returns
        -------
        left, right, d_left, d_right
            Массивы двух ветвей и их производные при `x=x_m`.
        """

        k = 2.0 * (float(energy) - self.v)

        left = integrate_from_left(
            k,
            self.h,
            self.match_index,
            seed=self.seed,
        )
        right = integrate_from_right(
            k,
            self.h,
            self.match_index,
            seed=self.seed,
        )

        d_left = derivative_from_left(left, self.match_index, self.h)
        d_right = derivative_from_right(right, self.match_index, self.h)
        return left, right, d_left, d_right

    def mismatch(self, energy: float) -> float:
        """Вычислить нормированный вронскиан двух ветвей.

        Сырый вронскиан в точке сшивки равен

            W = psi_L' psi_R - psi_R' psi_L.

        При собственной энергии левая и правая ветви являются частями одного и
        того же решения и потому линейно зависимы. Следовательно, `W(E)=0`.

        Вронскиан делится на произведение локальных норм в плоскости
        `(psi, psi')`. Такая нормировка:

        * не меняет положение нулей;
        * делает результат независимым от произвольных seed-масштабов;
        * ограничивает типичный масштаб mismatch;
        * не имеет сингулярности, даже если `psi(x_m)=0`.
        """

        left, right, d_left, d_right = self._branches(energy)
        m = self.match_index

        raw = d_left * right[m] - d_right * left[m]
        scale_left = np.hypot(left[m], d_left)
        scale_right = np.hypot(right[m], d_right)
        denominator = scale_left * scale_right

        # Нулевой или нечисловой знаменатель означает, что хотя бы одна ветвь
        # численно выродилась в точке сшивки. Возвращаем NaN, чтобы процедуры
        # сканирования могли пропустить такую энергию.
        if denominator == 0.0 or not np.isfinite(denominator):
            return float("nan")

        return float(raw / denominator)

    @staticmethod
    def _bisect(
        function: Callable[[float], float],
        a: float,
        b: float,
        *,
        energy_tolerance: float,
        function_tolerance: float,
        max_iterations: int = 200,
    ) -> float:
        """Уточнить корень функции методом бисекции.

        Интервал `[a,b]` должен содержать смену знака. На каждой итерации он
        делится пополам, после чего сохраняется та половина, где знак меняется.
        Метод медленнее Ньютона, зато не требует производной и гарантированно
        сходится при непрерывной функции и корректном bracketing.

        Остановка происходит по одному из двух критериев:

        * mismatch в середине достаточно мал;
        * половина ширины энергетического интервала достаточно мала.
        """

        fa = function(a)
        fb = function(b)

        if not np.isfinite(fa) or not np.isfinite(fb):
            raise ValueError("Non-finite mismatch at a bracketing endpoint")
        if abs(fa) <= function_tolerance:
            return float(a)
        if abs(fb) <= function_tolerance:
            return float(b)
        if fa * fb > 0:
            raise ValueError("Bisection interval does not bracket a root")

        left, right = float(a), float(b)
        f_left = float(fa)

        for _ in range(max_iterations):
            middle = 0.5 * (left + right)
            f_middle = function(middle)

            if not np.isfinite(f_middle):
                raise FloatingPointError(
                    "Mismatch became non-finite during bisection"
                )

            if (
                abs(f_middle) <= function_tolerance
                or 0.5 * (right - left) <= energy_tolerance
            ):
                return float(middle)

            # Если знак меняется на левой половине, корень лежит там;
            # иначе сдвигаем левую границу в середину.
            if f_left * f_middle <= 0:
                right = middle
            else:
                left = middle
                f_left = f_middle

        # Если достигнут лимит итераций, возвращаем середину последнего
        # интервала как лучшую доступную оценку.
        return float(0.5 * (left + right))

    def _fd_sturm_count(self, energy: float) -> int:
        """Оценить число конечно-разностных уровней ниже заданной энергии.

        Для вспомогательной задачи используется трёхдиагональная матрица
        второго порядка, соответствующая оператору

            H = -1/2 d²/dx² + V(x)

        с условиями Дирихле. Последовательность Штурма позволяет узнать число
        собственных значений ниже `energy`, не вычисляя весь спектр матрицы.

        Важно: эта функция служит только для грубого отделения уровней друг от
        друга. Финальные энергии находятся методом Нумерова.
        """

        # После исключения граничных узлов диагональ гамильтониана равна
        # 1/h² + V_i, а внедиагональные элементы равны -1/(2h²).
        diagonal = (1.0 / self.h**2) + self.v[1:-1]
        off_diagonal_squared = 1.0 / (4.0 * self.h**4)
        tiny = 1.0e-14

        # Рекурсия LDL^T-разложения матрицы H-EI. Число отрицательных pivot
        # равно числу собственных значений ниже E.
        pivot = float(diagonal[0] - energy)
        if abs(pivot) < tiny:
            # Точный нулевой pivot привёл бы к делению на ноль. Малый
            # отрицательный сдвиг сохраняет правильный подсчёт при проходе
            # через собственное значение.
            pivot = -tiny
        count = 1 if pivot < 0.0 else 0

        for value in diagonal[1:]:
            pivot = float(value - energy - off_diagonal_squared / pivot)
            if abs(pivot) < tiny:
                pivot = -tiny
            if pivot < 0.0:
                count += 1

        return count

    def _fd_eigenvalue_estimates(
        self,
        energy_min: float,
        energy_max: float,
        n_states: int,
        *,
        tolerance: float = 1.0e-9,
    ) -> FloatArray:
        """Получить грубую оценку каждой энергии через счётчик Штурма.

        Для уровня с порядковым номером `target` бисекцией ищется место, где
        число собственных значений ниже энергии меняется с `target` на
        `target+1`. Полученные значения достаточно точны, чтобы выделить для
        каждого физического уровня отдельный интервал поиска Numerov-корня.
        """

        below_window = self._fd_sturm_count(energy_min)
        available = self._fd_sturm_count(energy_max) - below_window

        if available < n_states:
            raise RuntimeError(
                f"The requested window contains only about {available} bound states. "
                "Increase energy_max."
            )

        estimates: list[float] = []

        # `target` — абсолютный номер уровня вспомогательной FD-задачи.
        for target in range(below_window, below_window + n_states):
            left, right = float(energy_min), float(energy_max)

            for _ in range(100):
                middle = 0.5 * (left + right)

                # Пока ниже middle не больше target уровней, искомый переход
                # находится правее; иначе — левее.
                if self._fd_sturm_count(middle) <= target:
                    left = middle
                else:
                    right = middle

                if right - left <= tolerance:
                    break

            estimates.append(0.5 * (left + right))

        return np.asarray(estimates, dtype=float)

    def _root_near_estimate(
        self,
        lower: float,
        upper: float,
        estimate: float,
        *,
        energy_tolerance: float,
        function_tolerance: float,
        local_scan_points: int = 24,
    ) -> float:
        """Найти Numerov-корень в окрестности FD-оценки.

        Сначала проверяются концы выделенного интервала. Если mismatch меняет
        знак между ними, сразу запускается бисекция. Иногда внутри интервала
        находится более сложная структура и на концах знаки совпадают; тогда
        выполняется небольшой локальный scan, после чего выбирается смена знака,
        расположенная ближе всего к `estimate`.
        """

        f_lower = self.mismatch(lower)
        f_upper = self.mismatch(upper)

        if np.isfinite(f_lower) and abs(f_lower) <= function_tolerance:
            return float(lower)
        if np.isfinite(f_upper) and abs(f_upper) <= function_tolerance:
            return float(upper)

        if (
            np.isfinite(f_lower)
            and np.isfinite(f_upper)
            and f_lower * f_upper < 0.0
        ):
            return self._bisect(
                self.mismatch,
                lower,
                upper,
                energy_tolerance=energy_tolerance,
                function_tolerance=function_tolerance,
            )

        # Дополнительный локальный scan нужен, если на концах большого
        # интервала знак не изменился, хотя корень внутри существует.
        mesh = np.linspace(lower, upper, local_scan_points, dtype=float)
        values = np.array([self.mismatch(e) for e in mesh], dtype=float)
        candidates: list[tuple[float, float, float]] = []

        for i in range(len(mesh) - 1):
            fa, fb = values[i], values[i + 1]

            if not np.isfinite(fa) or not np.isfinite(fb):
                continue
            if abs(fa) <= function_tolerance:
                return float(mesh[i])
            if fa * fb < 0.0:
                centre = 0.5 * (mesh[i] + mesh[i + 1])
                # В первый элемент кортежа кладём расстояние до FD-оценки,
                # чтобы затем выбрать наиболее правдоподобный интервал.
                candidates.append(
                    (abs(centre - estimate), mesh[i], mesh[i + 1])
                )

        if not candidates:
            raise RuntimeError(
                "Could not bracket a Numerov mismatch root near the finite-difference "
                f"estimate E≈{estimate:.12g}. Increase grid resolution or adjust the "
                "energy window."
            )

        _, a, b = min(candidates, key=lambda item: item[0])
        return self._bisect(
            self.mismatch,
            float(a),
            float(b),
            energy_tolerance=energy_tolerance,
            function_tolerance=function_tolerance,
        )

    def scan_mismatch(
        self,
        energy_min: float,
        energy_max: float,
        *,
        scan_points: int = 1200,
    ) -> tuple[FloatArray, FloatArray]:
        """Вычислить mismatch на равномерной энергетической сетке.

        Эта функция используется для построения диагностического графика и как
        запасной способ поиска корней, когда число искомых состояний заранее не
        задано.

        Returns
        -------
        energies, values:
            Массив пробных энергий и соответствующий массив `Delta(E)`.
        """

        if energy_max <= energy_min:
            raise ValueError("energy_max must be greater than energy_min")
        if scan_points < 20:
            raise ValueError("scan_points must be at least 20")

        energies = np.linspace(
            energy_min,
            energy_max,
            scan_points,
            dtype=float,
        )
        values = np.array([self.mismatch(e) for e in energies], dtype=float)
        return energies, values

    def find_eigenvalues(
        self,
        energy_min: float,
        energy_max: float,
        *,
        n_states: int | None = None,
        scan_points: int = 1200,
        energy_tolerance: float = 1.0e-11,
        function_tolerance: float = 1.0e-10,
    ) -> FloatArray:
        """Найти собственные энергии как нули Numerov mismatch-функции.

        Есть два режима.

        `n_states` задан:
            Сначала последовательность Штурма даёт одну грубую оценку на
            уровень, затем каждый Numerov-корень уточняется отдельно. Это
            основной и наиболее надёжный режим, особенно для почти вырожденных
            туннельных дублетов.

        `n_states is None`:
            Выполняется глобальный равномерный scan и ищутся все смены знака.
            Этот режим проще, но способен пропустить два очень близких корня.
        """

        if energy_max <= energy_min:
            raise ValueError("energy_max must be greater than energy_min")

        if n_states is not None:
            if n_states < 1:
                raise ValueError("n_states must be positive")

            estimates = self._fd_eigenvalue_estimates(
                energy_min,
                energy_max,
                n_states,
            )

            # Границы индивидуальных интервалов ставятся посередине между
            # соседними FD-оценками. Первый и последний интервалы ограничены
            # пользовательским энергетическим окном.
            boundaries = np.empty(n_states + 1, dtype=float)
            boundaries[0] = energy_min
            boundaries[-1] = energy_max
            if n_states > 1:
                boundaries[1:-1] = 0.5 * (
                    estimates[:-1] + estimates[1:]
                )

            roots = [
                self._root_near_estimate(
                    float(boundaries[i]),
                    float(boundaries[i + 1]),
                    float(estimates[i]),
                    energy_tolerance=energy_tolerance,
                    function_tolerance=function_tolerance,
                )
                for i in range(n_states)
            ]
            return np.asarray(roots, dtype=float)

        # Запасной режим: ищем все смены знака mismatch на общей сетке.
        energies, values = self.scan_mismatch(
            energy_min,
            energy_max,
            scan_points=scan_points,
        )

        roots: list[float] = []
        for i in range(len(energies) - 1):
            a, b = float(energies[i]), float(energies[i + 1])
            fa, fb = float(values[i]), float(values[i + 1])

            if not np.isfinite(fa) or not np.isfinite(fb):
                continue

            if abs(fa) <= function_tolerance:
                root = a
            elif fa * fb < 0.0:
                root = self._bisect(
                    self.mismatch,
                    a,
                    b,
                    energy_tolerance=energy_tolerance,
                    function_tolerance=function_tolerance,
                )
            else:
                continue

            # Один и тот же корень может попасть в соседние интервалы из-за
            # конечной точности. Не добавляем почти совпадающие дубликаты.
            if not roots or abs(root - roots[-1]) > 10.0 * energy_tolerance:
                roots.append(root)

        return np.asarray(roots, dtype=float)

    def build_eigenstate(
        self,
        energy: float,
        *,
        index: int = 0,
    ) -> Eigenstate:
        """Сшить и нормировать волновую функцию для найденной энергии.

        При точной собственной энергии векторы

            (psi_L, psi_L') и (psi_R, psi_R')

        в точке сшивки пропорциональны. Из-за конечной численной погрешности
        пропорциональность не идеальна, поэтому масштаб правой ветви выбирается
        методом наименьших квадратов в этой двумерной плоскости.
        """

        left, right, d_left, d_right = self._branches(energy)
        m = self.match_index

        # Ищем коэффициент scale, минимизирующий квадрат расстояния между
        # (psi_L, psi_L') и scale*(psi_R, psi_R'). Такая формула остаётся
        # корректной даже при узле psi(x_m)=0.
        denominator = right[m] ** 2 + d_right**2
        if denominator == 0.0:
            raise FloatingPointError(
                "Right branch vanished at the matching point"
            )

        scale = (
            left[m] * right[m] + d_left * d_right
        ) / denominator
        right_scaled = scale * right

        # Собираем единый массив: слева берём левую ветвь, справа —
        # масштабированную правую. В самой точке сшивки усредняем два значения,
        # чтобы не отдавать предпочтение одной стороне при малом рассогласовании.
        psi = np.empty_like(self.x)
        psi[: m + 1] = left[: m + 1]
        psi[m + 1 :] = right_scaled[m + 1 :]
        psi[m] = 0.5 * (left[m] + right_scaled[m])

        # Нормировка: интеграл |psi|^2 dx должен быть равен единице.
        # В новых версиях NumPy функция называется `trapezoid`, в старых —
        # `trapz`; getattr сохраняет совместимость с обеими версиями.
        integrate = getattr(np, "trapezoid", np.trapz)
        norm2 = float(integrate(psi**2, self.x))

        if norm2 <= 0.0 or not np.isfinite(norm2):
            raise FloatingPointError(
                "Could not normalize the matched wave function"
            )

        psi /= np.sqrt(norm2)

        # Глобальный знак физически не наблюдаем. Для стабильных графиков и
        # сравнений выбираем соглашение: в точке максимального |psi| значение
        # должно быть положительным.
        peak = int(np.argmax(np.abs(psi)))
        if psi[peak] < 0:
            psi *= -1.0

        residual = abs(self.mismatch(energy))

        return Eigenstate(
            index=int(index),
            energy=float(energy),
            x=self.x.copy(),
            psi=psi,
            residual=float(residual),
        )

    def solve(
        self,
        energy_min: float,
        energy_max: float,
        *,
        n_states: int,
        scan_points: int = 1200,
        energy_tolerance: float = 1.0e-11,
        function_tolerance: float = 1.0e-10,
    ) -> list[Eigenstate]:
        """Выполнить полный расчёт энергий и волновых функций.

        Это главный пользовательский метод класса. Сначала он вызывает
        `find_eigenvalues`, затем для каждой энергии вызывает
        `build_eigenstate` и возвращает готовый список объектов `Eigenstate`.
        """

        eigenvalues = self.find_eigenvalues(
            energy_min,
            energy_max,
            n_states=n_states,
            scan_points=scan_points,
            energy_tolerance=energy_tolerance,
            function_tolerance=function_tolerance,
        )

        return [
            self.build_eigenstate(energy, index=i)
            for i, energy in enumerate(eigenvalues)
        ]
