"""第09讲 转化漏斗与分端比较
业务问题：哪一步流失最多，分母到底是什么？
全部使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)

# %% 会话级、严格嵌套的漏斗
s=load_tables()['sessions'];stages=['viewed_product','added_cart','checkout','placed_order','paid_order']
counts=np.r_[len(s),s[stages].sum().to_numpy()]
labels=['Sessions','Viewed product','Added cart','Checkout','Order placed','Paid order']
funnel=pd.DataFrame({'stage':labels,'sessions':counts})
funnel['overall_rate']=funnel.sessions/len(s)
funnel['step_rate']=funnel.sessions/funnel.sessions.shift(1)
assert np.all(np.diff(counts)<=0)
show(funnel);save_table(funnel,9,'session_funnel')

# %% 分端比较仍以各端会话数为分母
device=s.groupby('device',as_index=False).agg(sessions=('session_id','nunique'),paid_sessions=('paid_order','sum'))
device['session_conversion']=device.paid_sessions/device.sessions
show(device);save_table(device,9,'device_conversion')
fig,ax=plt.subplots(figsize=(8,4));ax.barh(labels[::-1],counts[::-1],color=BLUE)
ax.set(title='Session-based conversion funnel',xlabel='Number of sessions');save_figure(fig,9,'conversion_funnel')
user_rate=s.groupby('customer_id').paid_order.max().mean()
print(f'会话支付转化率={s.paid_order.mean():.2%}；用户支付比例={user_rate:.2%}，分母不同。')
