import { useEffect, useMemo, useState } from "react";
import { formatCustomers } from "../utils/forecast";

type ForecastRow = {
  day: number;
  targetDate: string;
  prediction: number | null;
};

type ForecastTrendChartProps = {
  rows: ForecastRow[];
};

type ChartPoint = {
  x: number;
  y: number;
  value: number;
  day: number;
  targetDate: string;
};

const VIEWBOX_WIDTH = 300;
const VIEWBOX_HEIGHT = 72;
const CHART_LEFT = 12;
const CHART_RIGHT = VIEWBOX_WIDTH - 4;
const CHART_TOP = 6;
const CHART_BOTTOM = 58;
const BASELINE_Y = CHART_BOTTOM;

function toChartPoints(rows: ForecastRow[]): ChartPoint[] {
  const validRows = rows.filter((row): row is ForecastRow & { prediction: number } => row.prediction !== null);
  if (validRows.length < 2) {
    return [];
  }

  const values = validRows.map((row) => row.prediction);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const range = Math.max(maxValue - minValue, 1);

  return validRows.map((row, index) => {
    const x =
      CHART_LEFT +
      (index * (CHART_RIGHT - CHART_LEFT)) / (validRows.length - 1);
    const normalized = (row.prediction - minValue) / range;
    const y = BASELINE_Y - normalized * (BASELINE_Y - CHART_TOP);
    return { x, y, value: row.prediction, day: row.day, targetDate: row.targetDate };
  });
}

export function ForecastTrendChart({ rows }: ForecastTrendChartProps) {
  const points = useMemo(() => toChartPoints(rows), [rows]);
  const [activeDay, setActiveDay] = useState<number | null>(null);

  useEffect(() => {
    if (points.length < 2) {
      setActiveDay(null);
      return;
    }
    setActiveDay((current) => {
      if (current && points.some((point) => point.day === current)) {
        return current;
      }
      return points[points.length - 1].day;
    });
  }, [points]);

  if (points.length < 2) {
    return (
      <div className="forecast-chart-shell flex min-h-[240px] items-center justify-center text-sm text-slate-500">
        Trend chart will appear when at least two forecast points are available.
      </div>
    );
  }

  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");
  const firstPoint = points[0];
  const lastPoint = points[points.length - 1];
  const areaPath = `${linePath} L ${lastPoint.x} ${BASELINE_Y} L ${firstPoint.x} ${BASELINE_Y} Z`;
  const activePoint =
    points.find((point) => point.day === activeDay) ?? points[points.length - 1];
  const minValue = Math.min(...points.map((point) => point.value));
  const maxValue = Math.max(...points.map((point) => point.value));
  const valueRange = Math.max(maxValue - minValue, 1);
  const valueToY = (value: number) =>
    BASELINE_Y - ((value - minValue) / valueRange) * (BASELINE_Y - CHART_TOP);
  const step = points.length > 1 ? points[1].x - points[0].x : 4;
  const hoverBandWidth = Math.max(4, step);
  const guideValues = [maxValue, minValue + valueRange * 0.5, minValue];
  const horizontalGuides = guideValues.map((value) => ({
    y: valueToY(value),
    key: String(value),
    label: formatCustomers(Math.round(value)),
  }));

  function focusDay(day: number) {
    setActiveDay(day);
  }

  function handleChartKeyDown(event: React.KeyboardEvent<SVGSVGElement>) {
    if (!activePoint) {
      return;
    }

    if (event.key === "ArrowLeft") {
      event.preventDefault();
      const previous = points.find((point) => point.day === activePoint.day - 1);
      if (previous) {
        focusDay(previous.day);
      }
      return;
    }

    if (event.key === "ArrowRight") {
      event.preventDefault();
      const next = points.find((point) => point.day === activePoint.day + 1);
      if (next) {
        focusDay(next.day);
      }
      return;
    }

    if (event.key === "Home") {
      event.preventDefault();
      focusDay(firstPoint.day);
      return;
    }

    if (event.key === "End") {
      event.preventDefault();
      focusDay(lastPoint.day);
    }
  }

  return (
    <div className="forecast-chart-shell">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
        <span className="font-semibold text-slate-700">Forecast trend</span>
        <span className="whitespace-nowrap">Day +{firstPoint.day} to +{lastPoint.day}</span>
      </div>

      <div className="mb-3 grid gap-1 rounded-lg border border-slate-200/80 bg-white/80 px-3 py-2 text-xs sm:grid-cols-[auto_1fr_auto] sm:items-center sm:gap-2">
        <span className="whitespace-nowrap font-semibold text-slate-700">Day +{activePoint.day}</span>
        <span className="text-slate-600 sm:text-center">{activePoint.targetDate}</span>
        <span className="whitespace-nowrap font-semibold text-ink-900 sm:text-right">
          {formatCustomers(activePoint.value)} customers
        </span>
      </div>

      <svg
        viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
        preserveAspectRatio="none"
        className="h-[260px] w-full md:h-[300px]"
        role="img"
        aria-label="Forecast trend chart"
        tabIndex={0}
        onMouseLeave={() => setActiveDay(points[points.length - 1].day)}
        onKeyDown={handleChartKeyDown}
      >
        <defs>
          <linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#7fcdf2" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#7fcdf2" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {horizontalGuides.map((guide) => (
          <g key={guide.key}>
            <line
              x1={CHART_LEFT}
              y1={guide.y}
              x2={CHART_RIGHT}
              y2={guide.y}
              stroke="#deebf6"
              strokeWidth="0.4"
            />
            <text
              x={CHART_LEFT - 1}
              y={guide.y - 0.8}
              textAnchor="end"
              fontSize="2.3"
              fill="#64748b"
            >
              {guide.label}
            </text>
          </g>
        ))}

        <line
          x1={CHART_LEFT}
          y1={BASELINE_Y}
          x2={CHART_RIGHT}
          y2={BASELINE_Y}
          stroke="#dbe7f2"
          strokeWidth="0.6"
        />

        <path d={areaPath} fill="url(#forecastFill)" />
        <path
          d={linePath}
          fill="none"
          stroke="#2584c9"
          strokeWidth="1.3"
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />

        {points.map((point) => (
          <rect
            key={`hover-${point.day}`}
            x={point.x - hoverBandWidth / 2}
            y={CHART_TOP}
            width={hoverBandWidth}
            height={BASELINE_Y - CHART_TOP + 8}
            fill="transparent"
            role="button"
            tabIndex={0}
            aria-label={`Day +${point.day}: ${formatCustomers(point.value)} customers`}
            onMouseEnter={() => focusDay(point.day)}
            onClick={() => focusDay(point.day)}
            onTouchStart={() => focusDay(point.day)}
            onFocus={() => focusDay(point.day)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                focusDay(point.day);
              }
            }}
          />
        ))}

        {points.map((point) => (
          <circle
            key={point.day}
            cx={point.x}
            cy={point.y}
            r={point.day === activePoint.day ? "1.75" : "1.2"}
            fill="#ffffff"
            stroke="#1b5d9d"
            strokeWidth="0.8"
            vectorEffect="non-scaling-stroke"
          >
            <title>{`Day +${point.day}: ${formatCustomers(point.value)} customers`}</title>
          </circle>
        ))}

        <line
          x1={activePoint.x}
          y1={CHART_TOP}
          x2={activePoint.x}
          y2={BASELINE_Y}
          stroke="#7fcdf2"
          strokeWidth="0.5"
          strokeDasharray="1.5 1.5"
        />

        <text x={CHART_LEFT} y={67} fontSize="2.9" fill="#64748b">
          +{firstPoint.day}
        </text>
        <text
          x={(CHART_LEFT + CHART_RIGHT) / 2}
          y={67}
          textAnchor="middle"
          fontSize="2.9"
          fill="#64748b"
        >
          +{Math.floor((firstPoint.day + lastPoint.day) / 2)}
        </text>
        <text x={CHART_RIGHT} y={67} textAnchor="end" fontSize="2.9" fill="#64748b">
          +{lastPoint.day}
        </text>
      </svg>
    </div>
  );
}
