"""
高考数据爬虫与采集模块 (Project 5 v3.0.0)

支持爬取/采集方式:
1. requests + BeautifulSoup: 静态 HTML 网页
2. pdfplumber: PDF 表格
3. Selenium: 动态网页 (仅合法公开访问条件下使用)
4. pandas: Excel/CSV 文件
5. 正则匹配: 网页文本/招生章程
6. 人工导入模板: 无法自动爬取时的降级方案

爬取原则:
- 只采集公开数据
- 不绕过登录、验证码、反爬限制或付费权限
- 设置访问间隔(默认3秒)
- 保留数据来源链接和采集时间
- 遵守 robots.txt
- 当前无真实URL时使用配置占位，不伪造真实数据来源

注意: 所有示例 URL 均为占位示例，实际使用时必须替换为各省教育考试院、
阳光高考、高校官网等公开真实地址。模拟数据由 data_generator.py 统一管理。
"""

import re
import os
import time
import json
import sqlite3
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime


# ================================================================
# 通用基类 BaseCrawler
# ================================================================

class BaseCrawler:
    """高考数据采集器基类 (所有具体采集器继承此类)"""

    SOURCE_NAME = "unknown"
    TABLE_NAME = "unknown"
    REQUIRED_COLUMNS = []
    ALL_COLUMNS = []
    SOURCE_TYPE = "html"

    # 中文字段名映射(原始页面中文 -> 标准英文)
    COLUMN_MAP = {}

    def __init__(self, delay_seconds=3, timeout=30, data_version="v1.0"):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        self.delay = delay_seconds
        self.timeout = timeout
        self.data_version = data_version
        self.warnings = []
        self.source_registry = {}

    # ---- 请求与限速 ----

    def _respect_rate_limit(self):
        time.sleep(self.delay)

    def check_robots(self, base_url):
        """检查目标网站的 robots.txt"""
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            resp = self.session.get(robots_url, timeout=10)
            if resp.status_code == 200:
                disallowed = []
                for line in resp.text.split("\n"):
                    line = line.strip()
                    if line.lower().startswith("disallow:"):
                        path = line.split(":", 1)[1].strip()
                        if path:
                            disallowed.append(path)
                return {"status": "ok", "disallowed_paths": disallowed}
            return {"status": "no_robots_txt"}
        except Exception as e:
            return {"status": "check_failed", "error": str(e)}

    def is_allowed(self, url, robots_info=None):
        if robots_info is None or robots_info.get("status") != "ok":
            return True
        parsed = urlparse(url)
        path = parsed.path or "/"
        for disallowed in robots_info.get("disallowed_paths", []):
            if path.startswith(disallowed):
                return False
        return True

    # ---- HTML 解析 ----

    def fetch_html(self, url, max_retries=3, encoding=None):
        """获取静态 HTML 页面"""
        if not url:
            self._log_warning("URL为空，跳过请求")
            return None
        self._respect_rate_limit()
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                if encoding:
                    resp.encoding = encoding
                else:
                    resp.encoding = resp.apparent_encoding or "utf-8"
                if resp.status_code == 200:
                    self._record_source(url, url)
                    return resp.text
                elif resp.status_code == 404:
                    self._log_warning(f"404 Not Found: {url}")
                    return None
                elif resp.status_code == 403:
                    self._log_warning(f"403 Forbidden: {url} (可能需人工下载)")
                    return None
                else:
                    self._log_warning(f"HTTP {resp.status_code}, retry {attempt + 1}: {url}")
                    time.sleep(2 ** (attempt + 1))
            except requests.RequestException as e:
                self._log_warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** (attempt + 1))
        return None

    def parse_html_tables(self, html, table_index=None):
        """解析 HTML 页面中的表格"""
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return None
        dataframes = []
        indices = [table_index] if table_index is not None else range(len(tables))
        for idx in indices:
            if idx >= len(tables):
                continue
            table = tables[idx]
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            data = []
            for row in rows[1:]:
                cols = row.find_all(["td", "th"])
                row_data = [col.get_text(strip=True) for col in cols]
                if any(row_data):
                    data.append(row_data)
            if headers and data:
                df = pd.DataFrame(data, columns=headers)
                dataframes.append(df)
        if table_index is not None:
            return dataframes[0] if dataframes else None
        return dataframes if dataframes else None

    # ---- PDF 解析 ----

    def parse_pdf_tables(self, pdf_path, page_range=None):
        """使用 pdfplumber 解析 PDF 表格"""
        import pdfplumber
        all_tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = pdf.pages[page_range[0]:page_range[1]] if page_range else pdf.pages
                for page in pages:
                    tables = page.extract_tables()
                    for table in tables:
                        if table and any(any(cell for cell in row) for row in table):
                            clean = [row for row in table if any(cell for cell in row)]
                            if clean:
                                df = pd.DataFrame(clean[1:], columns=clean[0])
                                all_tables.append(df)
            self._record_source(pdf_path, pdf_path)
            return pd.concat(all_tables, ignore_index=True) if all_tables else None
        except Exception as e:
            self._log_warning(f"PDF parse failed: {e}")
            return None

    # ---- Excel/CSV 解析 ----

    def parse_excel(self, file_path, sheet_name=0):
        """读取 Excel 文件"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            self._record_source(file_path, file_path)
            return df
        except Exception as e:
            self._log_warning(f"Excel read failed: {e}")
            return None

    def parse_csv(self, file_path, encoding="utf-8"):
        """读取 CSV 文件"""
        for enc in [encoding, "gbk", "gb2312", "utf-8-sig"]:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                self._record_source(file_path, file_path)
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self._log_warning(f"CSV read failed ({enc}): {e}")
                return None
        return None

    # ---- 文本正则提取 ----

    def parse_text_by_regex(self, text, patterns, group_names=None):
        """通过正则表达式从文本中提取结构化数据"""
        if not text or not patterns:
            return {}
        results = {}
        for key, pattern_list in patterns.items():
            if isinstance(pattern_list, re.Pattern):
                pattern_list = [pattern_list]
            for pat in pattern_list:
                matches = pat.findall(text)
                if matches:
                    results[key] = matches[0] if len(matches) == 1 else matches
                    break
        return results

    # ---- 来源追溯 ----

    def add_source_trace(self, df, source_url="", source_name=None):
        """为 DataFrame 添加完整的来源追溯字段"""
        if df is None or df.empty:
            return df
        df = df.copy()
        df["source_url"] = source_url or ""
        df["source_name"] = source_name or self.SOURCE_NAME
        df["crawl_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df["data_version"] = self.data_version
        df["source_type"] = self.SOURCE_TYPE
        return df

    def _record_source(self, key, url):
        self.source_registry[key] = {
            "url": url,
            "crawl_time": datetime.now().isoformat(),
        }

    # ---- 字段校验 ----

    def validate_required_columns(self, df):
        """检查 DataFrame 是否包含所有必填字段"""
        if df is None or df.empty:
            return {"valid": False, "missing": self.REQUIRED_COLUMNS, "reason": "DataFrame为空"}
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            self._log_warning(f"缺少必填字段: {missing}，已用默认值填充")
        return {"valid": len(missing) == 0, "missing": missing}

    def validate_schema(self, df):
        """检查字段完整性并补默认值"""
        if df is None or df.empty:
            self._log_warning("DataFrame为空，schema校验失败")
            return pd.DataFrame(columns=self.ALL_COLUMNS) if self.ALL_COLUMNS else pd.DataFrame()

        for col in self.ALL_COLUMNS:
            if col not in df.columns:
                df[col] = "" if col.startswith("source_") or col.endswith("_name") or col.endswith("_url") else 0

        if self.COLUMN_MAP:
            df = df.rename(columns={v: k for k, v in self.COLUMN_MAP.items() if v in df.columns})
        return df

    # ---- 保存 ----

    def save_to_csv(self, df, filepath, encoding="utf-8-sig"):
        """保存为 CSV"""
        if df is None or df.empty:
            self._log_warning(f"DataFrame为空，跳过保存: {filepath}")
            return False
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        df.to_csv(filepath, index=False, encoding=encoding)
        print(f"[SAVED] {filepath} ({len(df)} rows)")
        return True

    def save_to_sqlite(self, df, db_path="gaokao_data.db", if_exists="replace"):
        """保存到 SQLite 数据库"""
        if df is None or df.empty:
            self._log_warning("DataFrame为空，跳过数据库写入")
            return False
        conn = sqlite3.connect(db_path)
        try:
            df.to_sql(self.TABLE_NAME, conn, if_exists=if_exists, index=False)
            print(f"[DB] 写入 {self.TABLE_NAME}: {len(df)} 行")
        finally:
            conn.close()
        return True

    # ---- 人工导入模板 ----

    def manual_import_template(self):
        """返回人工导入模板字段列表和说明

        当网站不允许自动爬取、PDF解析失败或需要人工校验时，
        可打印此模板，由人工按字段录入后通过 Excel/CSV 导入。

        Returns:
            dict: {字段: {类型, 说明, 示例}}
        """
        return {col: {"type": "string", "description": "", "example": ""} for col in self.ALL_COLUMNS}

    # ---- 日志 ----

    def _log_warning(self, msg):
        self.warnings.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        print(f"[WARN] {msg}")

    def get_warnings(self):
        return self.warnings

    # ---- URL 列表占位 ----

    def crawl_url_list(self):
        """
        返回待采集 URL 列表。

        当前无真实 URL 时返回空列表。
        真实部署时需替换为各省教育考试院、阳光高考等公开链接。
        """
        return []

    # ---- 完整流程 ----

    def run(self, urls=None):
        """
        串联采集全流程: crawl -> fetch -> parse -> build -> validate -> trace -> save
        """
        print(f"\n[{self.SOURCE_NAME}] 开始采集...")
        if urls is None:
            urls = self.crawl_url_list()

        all_dfs = []
        for url in urls:
            raw = self.fetch(url, url)
            if raw is None:
                continue
            parsed = self.parse_response(raw, url)
            if parsed is None or (isinstance(parsed, pd.DataFrame) and parsed.empty):
                continue
            all_dfs.append(parsed)

        if not all_dfs:
            self._log_warning("未采集到任何数据，请检查URL或使用手动导入")
            return pd.DataFrame(columns=self.ALL_COLUMNS)

        df = self.build_dataframe(all_dfs)
        df = self.validate_schema(df)
        df = self.add_source_trace(df)
        print(f"[{self.SOURCE_NAME}] 采集完成: {len(df)} 行")
        return df

    # ---- 子类需重写的方法 ----

    def fetch(self, url_or_path, context=None):
        """根据 context(文件路径或URL) 获取原始数据。子类按需重写。"""
        return self.fetch_html(url_or_path)

    def parse_response(self, raw, url_or_path=""):
        """解析原始响应为 DataFrame。子类必须重写。"""
        return self.parse_html_tables(raw)

    def build_dataframe(self, dfs_or_data):
        """将解析结果整理为标准 DataFrame。子类按需重写。"""
        if isinstance(dfs_or_data, pd.DataFrame):
            return dfs_or_data
        if isinstance(dfs_or_data, list):
            return pd.concat(dfs_or_data, ignore_index=True) if dfs_or_data else pd.DataFrame()
        return pd.DataFrame()


# 向后兼容别名
GaoKaoDataCrawler = BaseCrawler


# ================================================================
# 1. SegmentTableCrawler — 一分一段表
# ================================================================

class SegmentTableCrawler(BaseCrawler):
    """一分一段表、批次线、考生人数采集器"""

    SOURCE_NAME = "segment_table"
    TABLE_NAME = "segment_table"
    SOURCE_TYPE = "html"

    REQUIRED_COLUMNS = ["province", "year", "subject_type", "score", "cumulative_count"]

    ALL_COLUMNS = [
        "province", "year", "subject_type", "score",
        "segment_count", "cumulative_count", "rank", "percentile",
        "batch_line", "batch_line_type", "total_exam_count",
        "source_url", "source_name", "crawl_time", "data_version", "source_type",
    ]

    COLUMN_MAP = {
        "score": "分数", "segment_count": "本段人数", "cumulative_count": "累计人数",
        "rank": "位次", "batch_line": "批次线",
    }

    def crawl_url_list(self):
        # 占位示例，真实部署时替换为各省教育考试院官方链接
        # 例如: http://www.hebeea.edu.cn/2024/yifenyiduan.html
        return []

    def parse_response(self, raw, url_or_path=""):
        if isinstance(raw, pd.DataFrame):
            return raw
        if isinstance(raw, str) and raw.endswith(".pdf"):
            return self.parse_pdf_tables(raw)
        if isinstance(raw, str) and raw.endswith((".xlsx", ".xls")):
            return self.parse_excel(raw)
        return self.parse_html_tables(raw)

    def manual_import_template(self):
        return """
# 一分一段表人工导入模板
# 字段说明:
#   province: 省份名称(如 河北省)
#   year: 年份(如 2024)
#   subject_type: 科类(物理类/历史类)
#   score: 分数(0-750)
#   segment_count: 本段人数
#   cumulative_count: 累计人数
#   rank: 位次
#   batch_line: 批次线分数

示例CSV格式:
province,year,subject_type,score,segment_count,cumulative_count,rank,batch_line
河北省,2024,物理类,750,10,10,10,432
河北省,2024,物理类,749,5,15,15,432
"""


# ================================================================
# 2. AdmissionLineCrawler — 院校投档线
# ================================================================

class AdmissionLineCrawler(BaseCrawler):
    """历年院校投档线采集器"""

    SOURCE_NAME = "admission_line"
    TABLE_NAME = "school_admission_line"
    SOURCE_TYPE = "html"

    REQUIRED_COLUMNS = ["province", "year", "school_code", "min_admission_score"]

    ALL_COLUMNS = [
        "province", "year", "batch", "subject_type",
        "school_code", "school_name", "major_group_code",
        "min_admission_score", "min_admission_rank",
        "plan_count", "admission_count",
        "source_url", "source_name", "crawl_time", "data_version", "source_type",
    ]

    COLUMN_MAP = {
        "school_code": "院校代码", "school_name": "院校名称",
        "major_group_code": "专业组代码",
        "min_admission_score": "最低投档分", "min_admission_rank": "最低投档位次",
        "plan_count": "计划数", "admission_count": "录取数",
    }

    def crawl_url_list(self):
        return []

    def parse_response(self, raw, url_or_path=""):
        if isinstance(raw, pd.DataFrame):
            return raw
        if isinstance(raw, str) and raw.endswith(".pdf"):
            return self.parse_pdf_tables(raw)
        return self.parse_html_tables(raw)

    def manual_import_template(self):
        return """
# 院校投档线人工导入模板
# 字段: province, year, batch, subject_type, school_code, school_name,
#        major_group_code, min_admission_score, min_admission_rank,
#        plan_count, admission_count
"""


# ================================================================
# 3. MajorAdmissionCrawler — 专业录取数据
# ================================================================

class MajorAdmissionCrawler(BaseCrawler):
    """历年专业录取数据采集器"""

    SOURCE_NAME = "major_admission"
    TABLE_NAME = "major_admission"
    SOURCE_TYPE = "html"

    REQUIRED_COLUMNS = ["province", "year", "school_code", "major_code", "min_admission_score"]

    ALL_COLUMNS = [
        "school_code", "school_name", "major_code", "major_name",
        "major_group_code", "province", "year",
        "min_admission_score", "min_admission_rank",
        "avg_score", "max_score",
        "plan_count", "admission_count",
        "subject_requirement", "adjustment_rule",
        "source_url", "source_name", "crawl_time", "data_version", "source_type",
    ]

    COLUMN_MAP = {
        "school_code": "院校代码", "school_name": "院校名称",
        "major_code": "专业代码", "major_name": "专业名称",
        "major_group_code": "专业组代码",
        "min_admission_score": "最低分", "min_admission_rank": "最低位次",
        "avg_score": "平均分", "max_score": "最高分",
        "plan_count": "计划数", "admission_count": "录取数",
        "subject_requirement": "选科要求", "adjustment_rule": "调剂规则",
    }

    def crawl_url_list(self):
        return []

    def parse_response(self, raw, url_or_path=""):
        if isinstance(raw, pd.DataFrame):
            return raw
        if isinstance(raw, str) and raw.endswith(".pdf"):
            return self.parse_pdf_tables(raw)
        return self.parse_html_tables(raw)

    def manual_import_template(self):
        return """
# 专业录取数据人工导入模板
# 字段: school_code, school_name, major_code, major_name,
#        major_group_code, province, year, min_admission_score,
#        min_admission_rank, avg_score, max_score, plan_count,
#        admission_count, subject_requirement, adjustment_rule
"""


# ================================================================
# 4. EnrollmentPlanCrawler — 招生计划
# ================================================================

class EnrollmentPlanCrawler(BaseCrawler):
    """招生计划、选科要求、学费、学制、特殊限制采集器"""

    SOURCE_NAME = "enrollment_plan"
    TABLE_NAME = "admission_plan"
    SOURCE_TYPE = "html"

    REQUIRED_COLUMNS = ["province", "year", "school_code", "major_code", "plan_count"]

    ALL_COLUMNS = [
        "school_code", "school_name", "major_group_code",
        "major_code", "major_name",
        "province", "year", "plan_count",
        "tuition", "duration",
        "subject_requirement",
        "is_sino_foreign", "is_normal_major", "is_medical_major",
        "single_subject_limit", "physical_limit", "remark",
        "source_url", "source_name", "crawl_time", "data_version", "source_type",
    ]

    COLUMN_MAP = {
        "school_code": "院校代码", "school_name": "院校名称",
        "major_group_code": "专业组代码",
        "major_code": "专业代码", "major_name": "专业名称",
        "plan_count": "计划数", "tuition": "学费", "duration": "学制",
        "subject_requirement": "选科要求",
        "single_subject_limit": "单科限制", "physical_limit": "身体条件",
        "remark": "备注",
    }

    def crawl_url_list(self):
        return []

    def parse_response(self, raw, url_or_path=""):
        if isinstance(raw, pd.DataFrame):
            return raw
        if isinstance(raw, str) and raw.endswith(".pdf"):
            return self.parse_pdf_tables(raw)
        if isinstance(raw, str) and raw.endswith((".xlsx", ".xls")):
            return self.parse_excel(raw)
        return self.parse_html_tables(raw)

    def manual_import_template(self):
        return """
# 招生计划人工导入模板
# 字段: school_code, school_name, major_group_code, major_code,
#        major_name, province, year, plan_count, tuition, duration,
#        subject_requirement, is_sino_foreign, is_normal_major,
#        is_medical_major, single_subject_limit, physical_limit, remark
"""


# ================================================================
# 5. MajorEmploymentCrawler — 专业就业数据
# ================================================================

class MajorEmploymentCrawler(BaseCrawler):
    """专业就业数据采集器(高校就业质量报告/招聘网站/统计年鉴)"""

    SOURCE_NAME = "major_employment"
    TABLE_NAME = "major_employment"
    SOURCE_TYPE = "pdf"

    REQUIRED_COLUMNS = ["major_code", "data_year"]

    ALL_COLUMNS = [
        "major_code", "major_name",
        "employment_rate", "postgraduate_rate",
        "civil_service_post_count",
        "average_salary", "median_salary",
        "job_count", "job_growth_rate",
        "industry_distribution", "main_employment_city",
        "industry_growth_score", "stability_score",
        "sentiment_warning_score",
        "data_year", "sample_size",
        "source_url", "source_name", "crawl_time", "data_version", "source_type",
    ]

    COLUMN_MAP = {
        "major_code": "专业代码", "major_name": "专业名称",
        "employment_rate": "就业率", "postgraduate_rate": "升学率",
        "civil_service_post_count": "考公岗位数",
        "average_salary": "平均薪资", "median_salary": "中位数薪资",
        "job_count": "招聘岗位数", "job_growth_rate": "岗位增长率",
        "industry_growth_score": "行业成长性", "stability_score": "稳定度",
        "sentiment_warning_score": "舆情预警",
    }

    def crawl_url_list(self):
        # 就业质量报告通常以PDF形式发布
        # 招聘数据可通过合法公开API获取（如招聘网站公开报告）
        # 当前为占位，真实部署需维护公开URL表
        return []

    def parse_response(self, raw, url_or_path=""):
        if isinstance(raw, pd.DataFrame):
            return raw
        if isinstance(raw, str) and raw.endswith(".pdf"):
            return self.parse_pdf_tables(raw)
        if isinstance(raw, str) and raw.endswith((".xlsx", ".xls")):
            return self.parse_excel(raw)
        return self.parse_html_tables(raw)

    def manual_import_template(self):
        return """
# 专业就业数据人工导入模板
# 注意: sentiment_warning_score 仅作为辅助预警字段，不作为核心就业评价依据
# 字段: major_code, major_name, employment_rate, postgraduate_rate,
#        civil_service_post_count, average_salary, median_salary,
#        job_count, job_growth_rate, industry_growth_score, stability_score,
#        sentiment_warning_score, data_year, sample_size
"""


# ================================================================
# 6. CityDataCrawler — 城市产业数据
# ================================================================

class CityDataCrawler(BaseCrawler):
    """城市产业、就业机会、薪资、生活成本采集器"""

    SOURCE_NAME = "city_data"
    TABLE_NAME = "city_data"
    SOURCE_TYPE = "html"

    REQUIRED_COLUMNS = ["city", "data_year"]

    ALL_COLUMNS = [
        "city", "province", "year",
        "gdp", "gdp_per_capita", "tertiary_industry_ratio",
        "key_industries",
        "high_tech_company_count", "listed_company_count",
        "related_job_count",
        "average_salary", "living_cost",
        "distance_from_home", "transport_score",
        "source_url", "source_name", "crawl_time", "data_version", "source_type",
    ]

    COLUMN_MAP = {
        "city": "城市", "gdp": "GDP", "gdp_per_capita": "人均GDP",
        "tertiary_industry_ratio": "第三产业占比",
        "key_industries": "重点产业",
        "high_tech_company_count": "高新企业数",
        "listed_company_count": "上市公司数",
        "average_salary": "平均薪资", "living_cost": "生活成本",
        "transport_score": "交通便利度",
    }

    def crawl_url_list(self):
        # 统计局、统计年鉴公开数据
        return []

    def parse_response(self, raw, url_or_path=""):
        if isinstance(raw, pd.DataFrame):
            return raw
        if isinstance(raw, str) and raw.endswith((".xlsx", ".xls")):
            return self.parse_excel(raw)
        return self.parse_html_tables(raw)

    def manual_import_template(self):
        return """
# 城市产业数据人工导入模板
# 字段: city, province, year, gdp, gdp_per_capita, tertiary_industry_ratio,
#        key_industries, high_tech_company_count, listed_company_count,
#        related_job_count, average_salary, living_cost,
#        distance_from_home, transport_score
"""


# ================================================================
# 7. CandidateProfileCollector — 考生画像
# ================================================================

class CandidateProfileCollector(BaseCrawler):
    """考生画像与家庭偏好采集器(前端问卷/人工录入/CSV导入，不涉及公开爬取)"""

    SOURCE_NAME = "candidate_profile"
    TABLE_NAME = "candidate_profile"
    SOURCE_TYPE = "manual"

    REQUIRED_COLUMNS = ["candidate_id", "province", "subject_type", "score", "rank"]

    ALL_COLUMNS = [
        "candidate_id", "province", "year", "subject_type",
        "score", "rank",
        "interest_direction", "strong_subjects",
        "excluded_majors", "preferred_cities",
        "family_budget", "risk_preference",
        "accept_adjustment", "accept_sino_foreign",
        "accept_far_city", "employment_first", "postgraduate_first",
        "create_time",
        "source_url", "source_name", "crawl_time", "data_version", "source_type",
    ]

    def crawl_url_list(self):
        # 考生画像由用户在前端填写，非公开爬取数据
        return []

    def parse_response(self, raw, url_or_path=""):
        """支持从JSON串或CSV导入"""
        if isinstance(raw, pd.DataFrame):
            return raw
        if isinstance(raw, dict):
            return pd.DataFrame([raw])
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return pd.DataFrame([data])
                if isinstance(data, list):
                    return pd.DataFrame(data)
            except json.JSONDecodeError:
                pass
            if raw.endswith(".csv"):
                return self.parse_csv(raw)
            if raw.endswith((".xlsx", ".xls")):
                return self.parse_excel(raw)
        return pd.DataFrame()

    def manual_import_template(self):
        return """
# 考生画像人工录入模板 (JSON格式示例)
{
  "candidate_id": "cand_2024_001",
  "province": "河北省",
  "year": 2024,
  "subject_type": "物理类",
  "score": 620,
  "rank": 8500,
  "interest_direction": ["计算机", "电子信息"],
  "strong_subjects": ["数学", "物理"],
  "excluded_majors": [],
  "preferred_cities": ["北京", "上海"],
  "family_budget": 20000,
  "risk_preference": "balanced",
  "accept_adjustment": 1,
  "accept_sino_foreign": 0,
  "accept_far_city": 1,
  "employment_first": 0,
  "postgraduate_first": 1
}
"""


# ================================================================
# 动态网页爬虫 (独立工具类，不继承 BaseCrawler)
# ================================================================

class DynamicPageCrawler:
    """动态网页爬虫(使用 Selenium + Chrome headless)

    仅用于合法公开可访问的动态页面。
    不绕过登录、验证码或任何安全机制。
    """

    def __init__(self, headless=True):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.driver = None

    def __enter__(self):
        from selenium import webdriver
        self.driver = webdriver.Chrome(options=self.options)
        return self

    def __exit__(self, *args):
        if self.driver:
            self.driver.quit()

    def fetch_page(self, url, wait_selector=None, wait_time=10):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        self.driver.get(url)
        if wait_selector:
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
            )
        return self.driver.page_source

    def fetch_dynamic_table(self, url, wait_selector="table", wait_time=10):
        html = self.fetch_page(url, wait_selector, wait_time)
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return None
        rows = tables[0].find_all("tr")
        if len(rows) < 2:
            return None
        headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        data = []
        for row in rows[1:]:
            cols = row.find_all(["td", "th"])
            row_data = [col.get_text(strip=True) for col in cols]
            if any(row_data):
                data.append(row_data)
        return pd.DataFrame(data, columns=headers) if data else None


# ================================================================
# 招生章程限制条件提取器
# ================================================================

class AdmissionRequirementExtractor:
    """从招生章程HTML/文本中提取选科、体检、单科、学费等限制条件"""

    PATTERNS = {
        "single_subject": [
            re.compile(r"(英语|数学|语文|物理|化学|生物|外语)单科成绩(不|须)?\s*(低于|达到|不低于|不少于|高于)\s*(\d+)\s*分"),
            re.compile(r"(英语|数学|语文)高考成绩\s*(须|应)?\s*(\d+)\s*分\s*以上"),
            re.compile(r"外语\s*(口试|口语)\s*(成绩|考试)\s*(须|应|要求)\s*(达到|通过)"),
        ],
        "physical": [
            re.compile(r"(不招|不宜报考|不录取|限报).*?(色盲|色弱|高度近视|乙肝)"),
            re.compile(r"身高\s*(不低于|要求|须达到)\s*(\d+)\s*(cm|厘米)"),
            re.compile(r"(裸眼视力|矫正视力)\s*(不低于|须达到)\s*([\d.]+)"),
        ],
        "subject_requirement": [
            re.compile(r"选考科目\s*要?求?\s*[：:]\s*(.+)"),
            re.compile(r"选科要求\s*[：:]\s*(.+)"),
            re.compile(r"首选科目\s*[：:]\s*(.+)"),
            re.compile(r"再选科目\s*[：:]\s*(.+)"),
            re.compile(r"必选\s*(物理|历史|化学|生物|政治|地理)"),
        ],
        "tuition": [
            re.compile(r"学费[：:]\s*(\d+[\d,]*)\s*元"),
            re.compile(r"收费标准[：:]\s*(\d+[\d,]*)\s*元"),
        ],
    }

    @classmethod
    def extract(cls, html_or_text):
        text = BeautifulSoup(html_or_text, "html.parser").get_text()
        results = {
            "single_subject_limits": [],
            "physical_limits": [],
            "subject_requirements": [],
            "tuition": None,
        }
        for pattern in cls.PATTERNS["single_subject"]:
            for match in pattern.finditer(text):
                results["single_subject_limits"].append(match.group(0))
        for pattern in cls.PATTERNS["physical"]:
            for match in pattern.finditer(text):
                results["physical_limits"].append(match.group(0))
        for pattern in cls.PATTERNS["subject_requirement"]:
            for match in pattern.finditer(text):
                results["subject_requirements"].append(match.group(0))
        for pattern in cls.PATTERNS["tuition"]:
            match = pattern.search(text)
            if match:
                try:
                    results["tuition"] = int(match.group(1).replace(",", ""))
                except ValueError:
                    pass
                break
        return results


# ================================================================
# 示例与演示
# ================================================================

def crawl_gaokao_example():
    """爬虫使用示例(使用占位URL，实际使用时替换为真实地址)"""
    crawler = GaoKaoDataCrawler(delay_seconds=3)
    base_url = "http://www.example-exam.edu.cn"
    robots_info = crawler.check_robots(base_url)
    print(f"Robots 检查: {robots_info}")
    url_example = f"{base_url}/2024/segment_physics.html"
    if crawler.is_allowed(url_example, robots_info):
        html = crawler.fetch_html(url_example)
        if html:
            tables = crawler.parse_html_tables(html)
            if tables and isinstance(tables, list):
                print(f"发现 {len(tables)} 个表格")
            elif isinstance(tables, pd.DataFrame):
                print(f"表格: {tables.shape}")
    else:
        print("robots.txt 禁止访问该 URL")

    # 演示各采集器的模板
    print("\n=== 人工导入模板示例 ===")
    seg = SegmentTableCrawler()
    print(seg.manual_import_template()[:200])

    print("\n=== 考生画像 JSON 模板 ===")
    cand = CandidateProfileCollector()
    print(cand.manual_import_template()[:300])


if __name__ == "__main__":
    crawl_gaokao_example()
