#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector
from utils import imgio as iio
import os

from IPython.display import display
import ipywidgets as widgets

import wx
import re
import shutil
import glob
from functools import partial


# %matplotlib Qt5
#plt.matplotlib.get_backend()

class ModalWin(wx.Dialog):
    def __init__(self, parent, dlg_class, pars=None):
        super().__init__(parent=parent, title='Flourophore analysis datasets')
        ####---- Variables
        self.SetEscapeId(12345)
        ####---- Widgets
        self.a = dlg_class(self, pars)
        self.buttonOk = wx.Button(self, wx.ID_OK, label='Next')
        ####---- Sizers
        self.sizerB = wx.StdDialogButtonSizer()
        self.sizerB.AddButton(self.buttonOk)
        
        self.sizerB.Realize()

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.a, border=10, flag=wx.EXPAND) #|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(self.sizerB, border=10, 
            flag=wx.EXPAND)  # |wx.ALIGN_RIGHT|wx.ALL)

        self.SetSizerAndFit(self.sizer)

        self.SetPosition(pt=(550, 200))


class PathPanel(wx.Panel):
    def __init__(self, parent, pars):
        super().__init__(parent=parent)
        ####---- Variables
        self.parent = parent
        ####---- Widgets
        label = ("1. Select Dir with datasets")
        self.text = wx.StaticText(self, label=label, pos=(10, 10))
        self.path = wx.TextCtrl(self, value='c:\\', pos=(10, 35))
        
        self.browse_btn = wx.Button(self, -1, "Browse", pos=(160, 35))
        self.Bind(wx.EVT_BUTTON, self.Browse, self.browse_btn)
        
        self.flr_lbl = wx.StaticText(self, label=f'Flurophore name:', pos=(10, 60))
        self.flr_name = wx.TextCtrl(self, value='', pos=(160, 60))

        
    def Browse(self, event=None):
        try:
            dlg = wx.DirDialog (None, "Choose dataset diretory", "", wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_OK:
                RootPath = dlg.GetPath()
                self.path.SetValue(RootPath)

            dlg.Destroy()
        
        except:
            pass
        
        
class DatasetProcConfigurator(wx.Panel):
    def __init__(self, parent, pars):
        super().__init__(parent=parent)
        ####---- Variables
        self.parent = parent
        self.root_dir, self.runs_info, self.base_wl = pars
        self.rows = []
        ####---- Widgets
        
        self.sizer = wx.FlexGridSizer(7)
        self.gen_titel()
        
        wl_idx = 0
        wl_step = 30
        for ds_dir, els in self.runs_info.items():
            ds_date, ds_times, ds_ress = els
            
            for ds_t, ds_res in zip(ds_times, ds_ress):
                wl = self.base_wl + wl_idx * wl_step
                if abs(wl-1045) < wl_step/2:
                    wl = 1045
                wl_idx += 1
                
                path = get_dir(f'{self.root_dir}/{ds_dir}', f'{ds_date}_Doc1_', ds_t)
                ch_map, n_chs, n_times = get_channels_times(path)
                self.gen_ds_row(ds_dir, ds_date, ds_t, n_chs, wl)
                
        
        self.SetSizerAndFit(self.sizer)
        
    def gen_ds_row(self, ds_dir, ds_date, ds_t, n_chs, wl):
        ds_process = wx.CheckBox(self)
        ds_process.SetValue(True)
        ds_dir_lbl = wx.StaticText(self, label=ds_dir)
        ds_date_lbl = wx.StaticText(self, label=ds_date)
        ds_t_lbl = wx.StaticText(self, label=ds_t)
        ds_nch_lbl = wx.StaticText(self, label=f'{n_chs:d}')
        ds_wl_lbl = wx.TextCtrl(self, value=f'{wl:d}')
        ds_lp = wx.TextCtrl(self, value='2')
        
        els = {
            'ds_process': ds_process,
            'ds_dir_lbl' : ds_dir_lbl,
            'ds_date_lbl' : ds_date_lbl,
            'ds_t_lbl' : ds_t_lbl,
            'ds_nch_lbl': ds_nch_lbl,
            'ds_wl_lbl' : ds_wl_lbl,
            'ds_lp': ds_lp
        }
        
        self.rows.append(els)
        
        self.sizer.Add(ds_process, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_dir_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_date_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_t_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_nch_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_wl_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_lp, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)

    def gen_titel(self):
        ds_process = wx.StaticText(self, label='process')
        ds_dir_lbl = wx.StaticText(self, label='directory')
        ds_date_lbl = wx.StaticText(self, label='date')
        ds_t_lbl = wx.StaticText(self, label='dataset')
        ds_nch_lbl = wx.StaticText(self, label='num channels')
        ds_wl_lbl = wx.StaticText(self, label='Wavelength, nm')
        ds_lp = wx.StaticText(self, label='Laser Power, %')

        self.sizer.Add(ds_process, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_dir_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_date_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_t_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_nch_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_wl_lbl, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)
        self.sizer.Add(ds_lp, border=10, flag=wx.EXPAND|wx.ALIGN_LEFT|wx.ALL)


def get_dir(root_dir, pref, doc, pos=None):
    if pos:
        return os.path.join(root_dir, pref+doc+'__pos_'+pos, '')
    else:
        return os.path.join(root_dir, pref+doc, '')


ch_names = [
    "[2BLUE]",
    "[5GREEN]",
    "[6RED]",
    "[7FarRED]",
    "[8FarFarRED]",
]

channel_names = ['Blue', 'Green', 'Red', 'Far Red', 'Far Far Red']
        

def get_channels_times(path):
    list_of_files_path = glob.glob(path+'\*.tif')
    list_of_files = [os.path.basename(fp).split('.')[0] for fp in list_of_files_path]
    
    if len(list_of_files) == 0:
        return None
    idx_time = 0
    idx_ch = 0
    idx_chI = 0
    all_parts = list_of_files[0].split(' ')
    for p_idx, part in enumerate(all_parts):
        if '[' in part:
            idx_ch = p_idx
    
        if 'Time' in part:
            idx_time = p_idx

        if '_C' in part:
            idx_chI = p_idx
            
    #print(idx_ch, idx_time)
    
    channels = [f.split(' ')[idx_ch] for f in list_of_files]
    
    channelsI = [f.split(' ')[idx_chI] for f in list_of_files]
    channelsI = [f.split('_')[1] for f in channelsI]
    channelsI = [int(f.split('C')[1]) for f in channelsI]
    
    ch_id_map = {ch_idx:ch_n for ch_n, ch_idx in zip(channels, channelsI)}
    #print(ch_id_map)
    
    #channels = set(channels)
    times = [f.split(' ')[idx_time] for f in list_of_files]
    times = set(times)
    
    #print(times, channels)
    
    #ch_idxs = []
    #for ch_idx, ch in enumerate(ch_names):
    #    if ch in channels:
    #        ch_idxs.append(ch_idx)
            
    times = list(times)
    times = [int(t.split('Time')[1]) for t in times]
    
    n_t = np.max(times)+1
    n_ch =  np.max(channelsI)+1
    #print(times, ch_idxs, n_t)
    
    return ch_id_map, n_ch, n_t


def get_im_name(ds_dir, doc, ch_idx, t_idx):
    ch_name = ch_names[ch_idx]
    
    return os.path.join(ds_dir, 
                        doc+'_Doc1_PMT - PMT '+ ch_name + ' _C' + '%02d'%ch_idx+ '_Time Time' + '%04d'%t_idx + '.ome.tif'
                        )


def get_runs_info(run_root, dirs):
    res = [0.7731239092495636, 0.7731239092495636, 4.0]
    runs_info = {}

    for d in dirs:
        p = os.path.join(run_root, d, '*' )
        dss = glob.glob(p)

        dates = []
        times = []
        ress = []
        for ds_i in dss:
            bn = os.path.basename(ds_i)
            if '_bk' in bn or '_ALGN' in bn:
                print('skipping IGNORED dir', bn)
                continue

            if len(glob.glob(os.path.join(ds_i, '*tif'))) == 0:
                print('skipping EMPTY dir', bn)
                continue

            findres = re.findall('(.*)_Doc1_(.*)', bn)
            if len(findres)==0:
                continue
                
            date, t = findres[0]
            
            
            dates.append(date)
            times.append(t)
            ress.append(res)

        dates = list(set(dates))
        assert len(dates)==1

        runs_info[d] = (dates[0], times, ress)

    return runs_info


def run():
    # print(plt.matplotlib.get_backend())
    app = wx.App()
    cont = True

    if cont:
        frameM = ModalWin(None, PathPanel)
        if frameM.ShowModal() == wx.ID_OK:
            #print("Exited by Ok button")
            pass
        else:
            #print("Exited by X button")
            cont = False
            
        path = frameM.a.path.Value
        flr_name = frameM.a.flr_name.Value
        frameM.Destroy()

    if cont:
        run_root = os.path.dirname(path)
        dirs = [os.path.basename(path)]
        
        #print(dirs, run_root)
        
        runs_info = get_runs_info(run_root, dirs)
        
        #print(runs_info)
        
        frameM = ModalWin(None, DatasetProcConfigurator, (run_root, runs_info, 800))
        if frameM.ShowModal() == wx.ID_OK:
            #print("Exited by Ok button")
            pass
        else:
            #print("Exited by X button")
            cont = False
            
        rows = frameM.a.rows
        
        #print(rows)
        
        ds_list = []
        for row in frameM.a.rows:
            process = row['ds_process'].Value
            run = row['ds_dir_lbl'].Label
            date = row['ds_date_lbl'].Label
            doc  = row['ds_t_lbl'].Label
            n_ch = int(row['ds_nch_lbl'].Label)
            
            wl = float(row['ds_wl_lbl'].Value)
            lp = float(row['ds_lp'].Value)
            
            if process:
                ds_list.append([run_root, run, date, doc, n_ch, wl, lp])
                
        n_chs = np.array([el[4] for el in ds_list])
        n_chs -= n_chs.min()
        assert n_chs.max() == 0, 'all datasets should have same number of channels'
        frameM.Destroy()
    # app.MainLoop()
    #print(ds_list)
    
    all_ds = load_ds(ds_list)
    all_ds_mip = {wl: ds.max(axis=1) for wl,ds in all_ds.items()}
    all_ds_lp = {wl:lp for run_root, run, date, doc, n_ch, wl, lp in ds_list}
    
    # return all_ds
    
    bbox = plot_ds_mip(all_ds_mip, all_ds_lp=all_ds_lp, fluorophore_name=flr_name, save_name=f'{run_root}\\{run}\\overview', select=True)
    #print(plt.matplotlib.get_backend())
    
    
    if bbox is not None:
        x,y,w,h = bbox
    
        # zoom: 
        all_ds_mip_z = {k:v[:, y:y+h, x:x+w] for k, v in all_ds_mip.items()}
        plot_ds_mip(all_ds_mip_z, all_ds_lp=all_ds_lp, fluorophore_name=flr_name, save_name=f'{run_root}\\{run}\\zoom')
    else:
        all_ds_mip_z = all_ds_mip
        

    plot_mean_intensity(all_ds_mip_z, fluorophore_name=flr_name, save_name=f'{run_root}\\{run}\\intensity_zoom')


def load_ds(ds_list):
    all_ds = {}
    for run_root, run, date, doc, n_ch, wl, lp in ds_list:
        path = get_dir(f'{run_root}\\{run}', f'{date}_Doc1_', doc)
        fnames = []
        for ch_idx in range(n_ch):
            fname = get_im_name(ds_dir=path, doc=doc, ch_idx=ch_idx, t_idx=0)
            fnames.append(fname)

        print(f'reading dataset {path}, wl={wl} nm')
        ds = [iio.read_mp_tiff(fn) for fn in fnames]
        ds = np.array(ds)

        all_ds[wl] = ds

    wl_sorted = sorted(all_ds)
    all_ds = {wl:all_ds[wl] for wl in wl_sorted}
    return all_ds


def plot_ds_mip(all_ds_mip, all_ds_lp, fluorophore_name, s=4, save_name=None, show=True, select=False):
    nx = len(all_ds_mip)
    all_wl = list(all_ds_mip.keys())
    c, h, w = all_ds_mip[all_wl[0]].shape
    
    ny = c
    s = 4

    peaks = []
    top = []
    
    hw_max = max(h, w)
    f_h = np.sqrt(h / hw_max)
    f_w = np.sqrt(w / hw_max)
    
    # f_h = (h / hw_max)
    # f_w = (w / hw_max)
    
    for do_log in [False, True]:
        fig, ax = plt.subplots(ny, nx, figsize=(nx*s*f_w, ny*s*f_h), sharey='all', sharex='all')

        for ch_idx in range(ny):
            all_ds = np.concatenate([ds[ch_idx].flatten() for ds in all_ds_mip.values()]).flatten()
            d_min, d_max = all_ds.min(), all_ds.max()
            if d_max == d_min:
                nums, bins = np.histogram(all_ds, bins=1000)
                # print(all_ds.shape, d_min, d_max)
                peak_idx = np.argmax(nums)
                peaks.append(bins[peak_idx])

                v_max = np.percentile(all_ds, 99.8)
                top.append(v_max)
            else:
                peaks.append(max(0, d_min-1))
                top.append(d_max+1)

        for i, (wl,ds) in enumerate(all_ds_mip.items()):
            for ch_idx, cds in enumerate(ds):
                if do_log:
                    im = np.log(cds+1)
                    ax[ch_idx][i].imshow(im, cmap='gray', vmin=np.log(peaks[ch_idx]+1), vmax=np.log(top[ch_idx]+1))
                else:
                    im = cds
                    ax[ch_idx][i].imshow(im, cmap='gray', vmin=peaks[ch_idx], vmax=top[ch_idx])

            wl = all_wl[i]
            lp = all_ds_lp[wl]
            
            ttl = f'{wl}nm\n{lp}%'
            ax[0][i].set_title(ttl, size=10)

        for i in range(ny):
            ax[i][0].set_ylabel(f'{channel_names[i]}', size=10)
            
        sfx = ', color log scale' if do_log else ''
        plt.suptitle(f'Fluorophore {fluorophore_name}{sfx}')
        #plt.tight_layout(pad=3.4, h_pad=0.2, w_pad=0.2)
        if save_name is not None:
            plt.savefig(fname=save_name + ('_log' if do_log else '') + '.png')
        
        bb = None
        bb0=[0,0,1,1]
                
        if show:
            if do_log and select:
                
                bb = bb0.copy()
                def onselect(eclick, erelease, bb):
                    # Extract the coordinates of the selected bounding box
                    x0, y0 = eclick.xdata, eclick.ydata
                    x1, y1 = erelease.xdata, erelease.ydata
                    bbox = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

                    # Print the bounding box coordinates
                    #print("Bounding Box:", bbox)

                    for i,v in enumerate(bbox):
                        bb[i] = int(v)

                onselect_setter = partial(onselect, bb=bb)
                selectors = []
                for i in range(ny):
                    for j in range(nx):
                        #if i!=0:
                        #    continue
                        ax_ij = ax[i][j]
                        #print(f'in select {i}{j}') 
                        selector = RectangleSelector(ax_ij, onselect_setter, 
                                                     drawtype='box', useblit=True, interactive=True)
                                                     
                        selectors.append(selector)

            plt.show()
            #print('out') 
        plt.close()
        
    if show and select:
        return bb if bb != bb0 else None


def plot_mean_intensity(all_ds_mip, fluorophore_name, s=4, save_name=None, show=True):
    all_wl = list(all_ds_mip.keys())
    c, h, w = all_ds_mip[all_wl[0]].shape
    
    ny = c
    
    means = np.array([[ all_ds_mip[wl][ch].mean() for wl in all_wl] for ch in range(ny)])

    fig, ax = plt.subplots(1, ny, figsize=(ny*s, s*0.6), sharey='all', sharex='all')

    for ch in range(ny):
        ax[ch].plot(all_wl, means[ch])
        ax[ch].set_title(channel_names[ch])
        ax[ch].set_xlabel('wavelength, nm')

    ax[0].set_ylabel('intensity, au')
    plt.suptitle(f'Fluorophore {fluorophore_name}')
    
    plt.tight_layout(pad=2, h_pad=0., w_pad=0.6)
    if save_name is not None:
        plt.savefig(fname=save_name + '.png')
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    all_ds = run()



# # ToDo:
# 1. +unify plots
# 2. +read in fluorophore name
# 3. +read in LP
# 4. +add croping XY
# 5. Save plots
# 6. build
# 7. Done
