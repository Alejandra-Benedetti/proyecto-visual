import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import EXPORTS_DIR


class ReportExporter:

    def export_excel(
        self,
        stats_df: pd.DataFrame,
        descriptions: list | None = None,
        detections_summary: list | None = None,
    ) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = EXPORTS_DIR / f"analisis_estadistico_{ts}.xlsx"

        with pd.ExcelWriter(str(out), engine="openpyxl") as writer:
            num_cols = stats_df.select_dtypes(include=[np.number]).columns
            stats_df[num_cols].describe().T.round(4).to_excel(
                writer, sheet_name="Resumen Estadístico"
            )
            stats_df.to_excel(writer, sheet_name="Datos Completos", index=False)

            if descriptions:
                names = (
                    stats_df["image_name"].tolist()
                    if "image_name" in stats_df.columns
                    else list(range(len(descriptions)))
                )
                pd.DataFrame({"imagen": names, "descripcion": descriptions}).to_excel(
                    writer, sheet_name="Descripciones IA", index=False
                )

            if detections_summary:
                pd.DataFrame(detections_summary).to_excel(
                    writer, sheet_name="Detecciones", index=False
                )

        return out

    def export_pdf(
        self,
        stats_df: pd.DataFrame,
        summary_text: str = "",
        chart_paths: list | None = None,
    ) -> Path:
        from fpdf import FPDF

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = EXPORTS_DIR / f"reporte_{ts}.pdf"

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Cover
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 22)
        pdf.ln(30)
        pdf.cell(0, 12, "ANÁLISIS ESTADÍSTICO DE IMÁGENES DIGITALES", align="C", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.ln(6)
        pdf.cell(0, 8, f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}", align="C", ln=True)
        pdf.cell(0, 8, f"Total imágenes analizadas: {len(stats_df)}", align="C", ln=True)

        # Summary
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 10, "Resumen Ejecutivo", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, summary_text or self._auto_summary(stats_df))

        # Key stats table
        key_cols = [
            c for c in [
                "mean_intensity", "std_intensity", "median_intensity",
                "entropy", "edge_density", "mean_saturation",
                "colorfulness", "texture_contrast",
            ] if c in stats_df.columns
        ]
        if key_cols:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "Estadísticas Descriptivas Clave", ln=True)
            pdf.set_font("Helvetica", "", 8)
            self._table(pdf, stats_df[key_cols].describe().round(3))

        # Charts
        for cp in (chart_paths or []):
            if Path(cp).exists():
                pdf.add_page()
                pdf.image(str(cp), x=10, y=30, w=190)

        pdf.output(str(out))
        return out

    @staticmethod
    def _auto_summary(df: pd.DataFrame) -> str:
        n = len(df)
        avg_i = df.get("mean_intensity", pd.Series([0])).mean()
        avg_e = df.get("entropy", pd.Series([0])).mean()
        return (
            f"El análisis comprende {n} imágenes digitales procesadas de manera automatizada. "
            f"La intensidad media del universo fue {avg_i:.1f} unidades, con entropía "
            f"promedio de {avg_e:.3f}. Se extrajeron variables estadísticas cuantificables "
            f"que permiten una caracterización objetiva y reproducible del conjunto de imágenes."
        )

    @staticmethod
    def _table(pdf, df: pd.DataFrame):
        col_w = 22
        row_h = 6
        pdf.set_fill_color(220, 230, 245)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(38, row_h, "Variable", border=1, fill=True)
        for col in df.columns:
            pdf.cell(col_w, row_h, str(col)[:8], border=1, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 7)
        for idx, row in df.iterrows():
            pdf.cell(38, row_h, str(idx)[:18], border=1)
            for val in row:
                text = f"{val:.3f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, row_h, text[:8], border=1)
            pdf.ln()
